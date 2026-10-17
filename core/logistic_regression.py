# core/logistic_regression.py

import joblib
from sklearn.linear_model import LogisticRegression
from core.model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Implémentation concrète de BaseModel avec la régression logistique.
    Hérite de BaseModel → doit implémenter fit, predict, save, load.
    """

    def __init__(self, **kwargs):
        # On crée un modèle scikit-learn vide (pas encore entraîné)
        # max_iter=1000 pour éviter les warnings de convergence
        self.model = LogisticRegression(max_iter=1000, **kwargs)

    def fit(self, X, y):
        """Entraîne la régression logistique."""
        self.model.fit(X, y)
        print("LogisticRegression : entraînement terminé.")
        return self

    def predict(self, X):
        """
        Retourne les PROBABILITÉS (entre 0 et 1) d'être infecté.
        On utilise predict_proba[:, 1] qui correspond à la classe '1' (infecté).
        """
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict_class(self, X, threshold=0.5):
        """
        Retourne directement la classe (0 ou 1) selon le seuil.
        Utile pour l'évaluation.
        """
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def save(self, filepath):
        """Sauvegarde le modèle entraîné sur le disque."""
        joblib.dump(self.model, filepath)
        print(f"Modèle sauvegardé : {filepath}")

    def load(self, filepath):
        """Charge un modèle entraîné depuis le disque."""
        self.model = joblib.load(filepath)
        print(f"Modèle chargé : {filepath}")
        return self