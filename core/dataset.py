
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class DatasetLoader:

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "patient_data.csv"
            )

        self.data_path = Path(data_path)

    def load_data(self):
        """Charger les données depuis le fichier CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {self.data_path}"
            )

        df = pd.read_csv(self.data_path)

        if df.empty:
            raise ValueError("Le dataset est vide.")

        return df

    def split_data(self, target="infected",
                   test_size=0.2, random_state=42):
        """Séparer les caractéristiques et la cible."""

        df = self.load_data()

        X = df.drop(columns=[target])
        y = df[target]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
