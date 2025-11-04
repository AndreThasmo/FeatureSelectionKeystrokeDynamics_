import os
import pandas as pd
from datetime import datetime
from app.data_loader import DataLoader
from app.feature_selector import FeatureSelector
from app.user_dataset_builder import UserDatasetBuilder
from app.model_trainer import ModelTrainer
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class KeystrokeEvaluator:
    """
    Classe responsável por avaliar dados de dinâmica de digitação utilizando diferentes métodos de seleção de atributos e classificadores.
    Ela carrega os dados, aplica seleção de atributos, treina modelos e registra métricas de avaliação.
    """

    selectors = [
        ('T-Score', 't_score_filter'),
        ('Fisher Score', 'fisher_score_filter'),#,
        ('Low Variance', 'low_variance_filter'),
        ('MDI', 'mdi_importance'),
        ('None', None)
    ]

    def __init__(
        self,
        csv_path: str,
        train_size=50,
        feature_counts: List[int] = [5, 10, 15],
        low_variance_threshold=0.01,
        verbosity: int = logging.INFO,
        random_state: int = 42
    ):
        """
        Inicializa o avaliador com o caminho para o dataset e os parâmetros.

        Args:
            csv_path (str): Caminho para o arquivo CSV contendo os dados de digitação.
            train_size (int): Número de amostras por usuário para o conjunto de treino.
            feature_counts (list): Lista com quantidades de atributos a serem selecionados para certos seletores.
            low_variance_threshold (float): Limite para filtrar atributos com baixa variância.
        """
        self.csv_path = csv_path
        self.train_size = train_size
        self.feature_counts = feature_counts
        self.low_variance_threshold = low_variance_threshold
        self.results = []
        self.random_state = random_state
        logging.getLogger().setLevel(verbosity)

    def _prepare_data(self) -> tuple[DataLoader, pd.DataFrame, dict]:
        """
        Prepara e carrega os dados do dataset original, codifica o alvo e separa por usuário.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.csv_path}")
        logger.info("Carregando e preparando dados...")
        loader = DataLoader(self.csv_path)
        data = loader.load_data()
        data = loader.encode_target(data)
        user_data_all = loader.split_by_user(data)
        return loader, data, user_data_all

    def _format_metrics(self, user: str, selector_name: str, selected_features: List[str], clf_name: str,
                        metrics: dict, train_time: float, test_time: float) -> dict:
        return {
            'User': user,
            'Feature Selection Algorithm': selector_name,
            'Number of Features': len(selected_features),
            'Selected Features': ', '.join(selected_features),
            'Classifier': clf_name,
            'Accuracy': metrics['accuracy'],
            'Balanced Accuracy': metrics['balanced_accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'Specificity': metrics['specificity'],
            'Confusion Matrix': str(metrics['confusion_matrix'].tolist()),
            'FPR': metrics.get('fpr'),
            'FNR': metrics.get('fnr'),
            'MCC': metrics.get('mcc'),
            'FNMR': metrics.get('fnmr'),
            'Train Time (s)': train_time,
            'Test Time (s)': test_time
        }

    def _store_result(self, result_dict: dict) -> None:
        """
        Armazena o resultado da avaliação de um classificador para um usuário.
        """
        self.results.append(result_dict)

    def _apply_selector_and_train(self, user: str, user_data_all: dict, selector_name: str, selected_features: List[str]) -> bool:
        """
        Aplica um conjunto de atributos selecionados para treino e avaliação do modelo.
        """
        if not selected_features:
            logger.warning(f"Nenhum atributo selecionado pelo seletor '{selector_name}' para o usuário '{user}'. Pulando treinamento.")
            return True

        selected_features_set = selected_features + ['target']

        user_subset = {
            k: v[[col for col in selected_features_set if col in v.columns]]
            for k, v in user_data_all.items()
        }
        user_builder = UserDatasetBuilder(user_subset, self.train_size, random_state=self.random_state)

        X_train_sel, y_train_sel = user_builder.get_training_data(user)
        X_test_sel, y_test_sel = user_builder.get_test_data(user)

        if X_train_sel.empty or X_test_sel.empty:
            logger.warning(f"Dados de treino ou teste vazios para o usuário '{user}' após seleção de atributos. Pulando treinamento.")
            return True
        if X_train_sel.shape[1] != len(selected_features):
            logger.warning(f"Dimensão inconsistente após seleção: {X_train_sel.shape[1]} colunas, esperadas {len(selected_features)}.")

        models = ModelTrainer.get_models()
        for clf_name, clf in models.items():
            logger.info(f"Treinando classificador {clf_name} com {len(selected_features)} atributos.")
            trainer = ModelTrainer(clf)
            train_time = trainer.train(X_train_sel, y_train_sel)
            y_pred, test_time = trainer.predict(X_test_sel)
            metrics = trainer.evaluate(y_test_sel, y_pred)

            result = self._format_metrics(user, selector_name, selected_features, clf_name,
                                          metrics, train_time, test_time)
            self._store_result(result)
        return True

    def _evaluate_user(self, user: str, user_data_all: dict) -> None:
        """
        Realiza o processo de avaliação por seletor e número de atributos para um usuário específico.
        """
        builder = UserDatasetBuilder(user_data_all, self.train_size, random_state=self.random_state, fix_user_sample=False)
        X_train, y_train = builder.get_training_data(user)
        X_test, y_test = builder.get_test_data(user)

        if X_train.empty or X_test.empty:
            logger.warning(f"Dados de treino ou teste estão vazios para o usuário '{user}'. Pulando avaliação.")
            return

        for selector_name, _ in self.selectors:
            logger.info(f"Aplicando seletor: {selector_name}")
            selector = FeatureSelector()

            if selector_name == 'None':
                clean_X = selector._ensure_numeric_frame(X_train)
                selected_features = [c for c in clean_X.columns if c != 'target']
                self._apply_selector_and_train(user, user_data_all, selector_name, selected_features)
                continue

            for n_features in self.feature_counts:
                if selector_name == 'Low Variance':
                    filtered_data = selector.low_variance_filter(train_data=X_train, top_n=n_features)
                    selected_features = [col for col in filtered_data.columns if col != 'target']
                elif selector_name == 'T-Score':
                    selected = selector.t_score_filter(X_train, y_train, top_n=n_features)
                    selected_features = [feat for feat, _ in selected]
                elif selector_name == 'MDI':
                    selected = selector.mdi_importance(X_train, y_train, top_n=n_features)
                    selected_features = [feat for feat, _ in selected]
                else:
                    selected = selector.fisher_score_filter(X_train, y_train, top_n=n_features)
                    selected_features = [feat for feat, _ in selected]

                if not selected_features:
                    logger.warning(f"Nenhum atributo retornado pelo seletor '{selector_name}' para o usuário '{user}'.")
                    continue

                self._apply_selector_and_train(user, user_data_all, selector_name, selected_features)

    def run(self, users: List[str] | None = None) -> None:
        """
        Executa a avaliação para todos os usuários definidos, com paralelismo.
        """
        import traceback
        loader, data, user_data_all = self._prepare_data()
        if users is None:
            users = list(user_data_all.keys())
        else:
            users = [user for user in users if user in user_data_all]

        logger.info(f"Iniciando avaliação para {len(users)} usuário(s).")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._evaluate_user, user, user_data_all): user
                for user in users
            }

            for future in as_completed(futures):
                user = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Erro ao avaliar o usuário {user}:\n{traceback.format_exc()}")

    def get_results(self):
        """
        Retorna os resultados da avaliação como um DataFrame do pandas.

        Returns:
            pd.DataFrame: DataFrame contendo os resultados da avaliação.
        """
        return pd.DataFrame(self.results)

    def save_results(self, output_dir="data/output") -> None:
        """
        Salva os resultados da avaliação em arquivos CSV e Excel com timestamp.

        Args:
            output_dir (str): Diretório onde os arquivos de resultado serão salvos.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(f"{output_dir}/keystroke_results_{timestamp}.csv", index=False)
        df.to_excel(f"{output_dir}/keystroke_results_{timestamp}.xlsx", index=False)
