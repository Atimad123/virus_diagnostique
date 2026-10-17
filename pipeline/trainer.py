# pipeline/trainer.py

import joblib
from pathlib import Path

from utils.preprocessing import Preprocessor
from core.logistic_regression import LogisticRegressionModel


class Trainer:
    """
    Responsable de l'entraînement complet :
    1. Prétraite les données
    2. Entraîne le modèle
    3. Sauvegarde le modèle ET le preprocessor
    """

    def __init__(self, model=None):
        # Si aucun modèle n'est fourni, on utilise la régression logistique par défaut
        self.model = model if model is not None else LogisticRegressionModel()
        self.preprocessor = Preprocessor()

    def train(self, X_train, y_train):
        """Prétraite puis entraîne le modèle."""
        print("\n=== ÉTAPE 1 : Prétraitement ===")
        X_train_scaled = self.preprocessor.fit_transform(X_train)

        print("\n=== ÉTAPE 2 : Entraînement ===")
        self.model.fit(X_train_scaled, y_train)

        print("\nEntraînement terminé.")
        return self

    def save(self, model_path="models/model.pkl",
             preprocessor_path="models/preprocessor.pkl"):
        """Sauvegarde le modèle et le preprocessor."""
        # Créer le dossier models/ s'il n'existe pas
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        self.model.save(model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor sauvegardé : {preprocessor_path}")