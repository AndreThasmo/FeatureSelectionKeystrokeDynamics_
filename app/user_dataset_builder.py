import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

IS_USER = 1
IS_IMPOSTER = 0

class UserDatasetBuilder:
    """
    Constrói datasets de treino e teste com amostras autênticas e impostoras para um dado usuário.
    """
    def __init__(self, user_data: dict[str, pd.DataFrame], train_size: int, random_state: int = 42, fix_user_sample: bool = True) -> None:
        """
        Inicializa o construtor com os dados de todos os usuários e o número de amostras de treino desejado.

        :param user_data: Dicionário contendo os dados de todos os usuários.
        :param train_size: Número de amostras a serem usadas para treino.
        :param random_state: Semente para geração de números aleatórios.
        :param fix_user_sample: Se True, seleciona as primeiras N amostras do usuário para treino; se False, seleciona aleatoriamente.
        """
        self.user_data = user_data
        self.train_size = train_size
        self.random_state = random_state
        self.fix_user_sample = fix_user_sample

    def get_training_data(self, user: str) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Retorna os dados de treino combinando amostras do usuário alvo (autênticas) e de outros usuários (impostoras).

        :param user: Identificador do usuário alvo.
        :return: Tupla (X_train, y_train) com os dados e rótulos de treino.
        """
        # Validação: checa se o usuário possui dados suficientes para treino
        if len(self.user_data[user]) <= self.train_size:
            raise ValueError(f"Usuário {user} não possui dados suficientes para treino.")
        # Seleciona as primeiras N amostras do usuário como dados positivos (autêntico) ou amostras aleatórias conforme fix_user_sample
        if self.fix_user_sample:
            train_user = self.user_data[user].iloc[:self.train_size]
        else:
            train_user = self.user_data[user].sample(n=self.train_size, random_state=self.random_state)

        # Seleciona amostras de outros usuários como impostores (negativo), garantindo mesma quantidade e colunas
        imposters = pd.concat([
            self.user_data[other].iloc[:self.train_size]
            for other in self.user_data if other != user
        ], ignore_index=True).sample(n=self.train_size, random_state=self.random_state)

        # Combina os dados positivos e negativos
        X_train = pd.concat([train_user, imposters], ignore_index=True)
        X_train = X_train.drop(columns=["target"], errors="ignore")
        y_train = np.hstack((
            np.ones(self.train_size) * IS_USER,         # Rótulo 1 para o usuário verdadeiro
            np.zeros(self.train_size) * IS_IMPOSTER     # Rótulo 0 para os impostores
        ))

        logger.info(f"Usuário {user} - Treinamento: {len(train_user)} autênticas, {len(imposters)} impostoras")

        return X_train, y_train

    def get_test_data(self, user: str) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Retorna os dados de teste combinando as amostras restantes do usuário alvo e dos impostores.

        :param user: Identificador do usuário alvo.
        :return: Tupla (X_test, y_test) com os dados e rótulos de teste.
        """
        # Validação: checa se o usuário possui dados suficientes para teste
        if len(self.user_data[user]) <= self.train_size:
            raise ValueError(f"Usuário {user} não possui dados suficientes para teste.")
        test_user = self.user_data[user].iloc[self.train_size:]
        imposters = pd.concat([
            self.user_data[other].iloc[self.train_size:]
            for other in self.user_data if other != user
        ])

        # Validações adicionais para garantir pelo menos uma amostra
        if test_user.empty:
            raise ValueError(f"Usuário {user} não possui dados suficientes para teste após a divisão.")
        if imposters.empty:
            raise ValueError("Não há dados suficientes de impostores para compor o conjunto de teste.")

        X_test = pd.concat([test_user, imposters], ignore_index=True)
        X_test = X_test.drop(columns=["target"], errors="ignore")
        y_test = np.hstack((np.ones(len(test_user)) * IS_USER, np.zeros(len(imposters)) * IS_IMPOSTER))

        logger.info(f"Usuário {user} - Teste: {len(test_user)} autênticas, {len(imposters)} impostoras")

        return X_test, y_test
