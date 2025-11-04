from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pandas import DataFrame

class ModelTrainer:
    """
    Classe responsável por treinar, prever e avaliar modelos de classificação supervisionada.
    """
    def __init__(self, model: Optional[Any] = None) -> None:
        """
        Inicializa o ModelTrainer com um modelo de classificação específico.

        Parâmetros:
            model: instância de um classificador do scikit-learn.
        """
        self.model = model

    def set_model(self, model: Any) -> None:
        """
        Define o modelo de classificação a ser utilizado.

        Parâmetros:
            model: instância de um classificador do scikit-learn.
        """
        self.model = model

    @staticmethod
    def get_models() -> Dict[str, Any]:
        """
        Retorna um dicionário com classificadores padrão disponíveis para experimentação.

        Retorno:
            dict: nomes dos modelos como chaves e instâncias dos modelos como valores.
        """
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, random_state=42),
            "KNN": KNeighborsClassifier()
        }

    def train(self, X_train: DataFrame, y_train: np.ndarray) -> float:
        """
        Treina o classificador fornecido com os dados de entrada.

        Parâmetros:
            X_train (array-like): matriz com atributos de treino.
            y_train (array-like): vetor com rótulos de treino.

        Retorno:
            float: tempo gasto em segundos para o treinamento.
        """
        if self.model is None:
            raise ValueError("Nenhum modelo foi definido. Use set_model() ou passe um modelo no construtor.")
        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()
        return end - start

    def predict(self, X_test: DataFrame) -> Tuple[np.ndarray, float]:
        """
        Realiza a predição dos rótulos com o modelo treinado.

        Parâmetros:
            X_test (array-like): matriz com atributos de teste.

        Retorno:
            tuple: predições e tempo gasto para previsão.
        """
        if self.model is None:
            raise ValueError("Nenhum modelo foi definido. Use set_model() ou passe um modelo no construtor.")
        start = time.time()
        y_pred = self.model.predict(X_test)
        end = time.time()
        return y_pred, end - start

    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Avalia o desempenho do modelo com base em diversas métricas.

        Parâmetros:
            y_test (array-like): rótulos reais.
            y_pred (array-like): rótulos preditos pelo modelo.

        Retorno:
            dict: métricas de avaliação como acurácia, f1, especificidade, MCC etc.
        """
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        labels = np.unique(y_test)
        report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

        tn, fp, fn, tp = conf_matrix.ravel()
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # = FMR
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # = FNMR
        fnmr = fnr
        mcc = matthews_corrcoef(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': conf_matrix,
            'report': report,
            'fpr': fpr,
            'fnr': fnr,
            'fnmr': fnmr,
            'mcc': mcc,
        }

    def predict_proba(self, X_test: DataFrame) -> np.ndarray:
        """
        Retorna as probabilidades preditas se o modelo suportar esse método.

        Parâmetros:
            X_test (array-like): matriz com atributos de teste.

        Retorno:
            array-like: probabilidades preditas.
        """
        if self.model is None:
            raise ValueError("Nenhum modelo foi definido. Use set_model() ou passe um modelo no construtor.")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)

        raise NotImplementedError("Este modelo não suporta probabilidade.")
