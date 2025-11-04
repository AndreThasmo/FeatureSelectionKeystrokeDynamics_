import warnings
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier

# Silenciar warnings de precisão numérica que ocorrem com colunas quase constantes
warnings.filterwarnings("ignore", category=RuntimeWarning)

class FeatureSelector:
    """
    Classe para seleção de atributos em dados de digitação por meio de diferentes filtros estatísticos.
    Permite a remoção de atributos com low-variance, seleção baseada em T-Score e Fisher Score,
    considerando dados específicos de usuários e impostores.
    """
    def __init__(self, columns_to_ignore: Optional[List[str]] = None):
        import os
        os.makedirs("logs", exist_ok=True)
        self.columns_to_ignore = columns_to_ignore or ['subject', 'sessionIndex', 'rep', 'target']

    def _ensure_numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dropa colunas ignoradas e tenta converter todas as demais para numérico.
        Colunas que não puderem ser convertidas viram NaN e são descartadas ao final.
        """
        df = df.drop(columns=self.columns_to_ignore, errors='ignore').copy()
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Mantém apenas colunas numéricas
        df = df.select_dtypes(include=[np.number])
        return df

    def low_variance_filter(self, train_data: pd.DataFrame, top_n: Optional[int] = None, variance_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Seleciona as colunas de maior variância, retornando as top_n se especificado, ou
        aplica um limiar de variância caso `variance_threshold` seja fornecido.
        """
        assert isinstance(train_data, pd.DataFrame), "train_data deve ser um DataFrame"
        train_data = self._ensure_numeric_frame(train_data)
        if train_data.empty:
            return train_data
        variances = train_data.var(numeric_only=True)
        variances = variances.fillna(0.0)
        if variance_threshold is not None:
            selected_columns = variances[variances > variance_threshold].index
            return train_data[selected_columns]
        sorted_variances = variances.sort_values(ascending=False)
        selected_columns = sorted_variances.head(top_n).index if top_n else sorted_variances.index
        return train_data[selected_columns]

    def t_score_filter(self, train_features: pd.DataFrame, y_train: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Calcula o T-Score para cada atributo com base em uma comparação binária (usuário vs impostores).
        """
        assert isinstance(train_features, pd.DataFrame), "train_features deve ser um DataFrame"
        assert len(np.unique(y_train)) == 2, "y_train deve conter apenas duas classes (usuário e impostor)"
        X = self._ensure_numeric_frame(train_features).reset_index(drop=True)
        if X.empty:
            return []
        y = pd.Series(y_train).reset_index(drop=True)
        mask = y == 1
        t_scores: dict[str, float] = {}
        for column in X.columns:
            x1 = X.loc[mask, column].astype(float).dropna()
            x0 = X.loc[~mask, column].astype(float).dropna()
            if len(x1) < 2 or len(x0) < 2:
                score = 0.0
            else:
                res = ttest_ind(x1, x0, nan_policy='omit', equal_var=False)
                stat_val = getattr(res, "statistic", res[0] if isinstance(res, (tuple, list, np.ndarray)) else np.nan)
                score = float(abs(stat_val)) # type: ignore
            if not np.isfinite(score):
                score = 0.0
            t_scores[column] = score
        pd.Series(t_scores).to_csv("logs/t_score_log.csv", index=True)
        sorted_scores = sorted(t_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def fisher_score_filter(self, train_features: pd.DataFrame, y_train: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Calcula o Fisher Score binário para cada atributo (maior separabilidade interclasse / menor variância intraclasse).
        """
        assert isinstance(train_features, pd.DataFrame), "train_features deve ser um DataFrame"
        assert len(np.unique(y_train)) == 2, "y_train deve conter apenas duas classes (usuário e impostor)"
        X = self._ensure_numeric_frame(train_features).reset_index(drop=True)
        if X.empty:
            return []
        y = pd.Series(y_train).reset_index(drop=True)
        mask = y == 1
        fisher_scores: dict[str, float] = {}
        for column in X.columns:
            x1 = X.loc[mask, column].astype(float).dropna()
            x0 = X.loc[~mask, column].astype(float).dropna()
            if len(x1) == 0 or len(x0) == 0:
                fisher_scores[column] = 0.0
                continue
            mu = pd.concat([x1, x0]).mean()
            mu1, mu0 = x1.mean(), x0.mean()
            var1, var0 = x1.var(ddof=1), x0.var(ddof=1)
            denominator = var1 + var0 # type: ignore
            if not np.isfinite(denominator) or denominator <= 0: # type: ignore
                fisher_scores[column] = 0.0
                continue
            numerator = (mu1 - mu) ** 2 + (mu0 - mu) ** 2
            score = float(numerator / denominator) # type: ignore
            if not np.isfinite(score):
                score = 0.0
            fisher_scores[column] = score
        pd.Series(fisher_scores).to_csv("logs/fisher_score_log.csv", index=True)
        sorted_scores = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def mdi_importance(self, train_features: pd.DataFrame, y_train: np.ndarray, top_n: Optional[int] = 10) -> List[Tuple[str, float]]:
        """
        Calcula a importância MDI (Mean Decrease in Impurity) usando um RandomForest real.
        """
        X = self._ensure_numeric_frame(train_features).reset_index(drop=True)
        X = X.fillna(X.median(numeric_only=True))
        if X.empty:
            return []
        y = pd.Series(y_train).reset_index(drop=True).values
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X, y) # type: ignore
        importances = rf.feature_importances_
        features = X.columns.tolist()
        sorted_scores = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n] if top_n else sorted_scores