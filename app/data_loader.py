import pandas as pd
import numpy as np

class DataLoader:
    """
    Classe responsável por carregar e preparar os dados do dataset de keystroke dynamics.
    """
    def __init__(self, path: str = 'data/DSL-StrongPasswordData.csv') -> None:
        """
        Inicializa o DataLoader com o caminho padrão do arquivo CSV.
        """
        self.path = path

    def load_data(self) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV especificado e retorna um DataFrame.
        """
        return pd.read_csv(self.path)

    def encode_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica a coluna 'subject' em valores numéricos e adiciona uma nova coluna 'target'.
        """
        data['target'] = data['subject'].astype('category').cat.codes
        return data

    def split_by_user(self, data: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """
        Separa os dados numéricos por usuário com base na coluna 'target' e retorna um dicionário.
        """
        numeric_data = data.select_dtypes(include=[np.number])
        return {user: numeric_data[numeric_data['target'] == user] for user in numeric_data['target'].unique()}
