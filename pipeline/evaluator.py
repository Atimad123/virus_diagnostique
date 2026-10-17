# pipeline/evaluator.py

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


class Evaluator:
    """Responsable de l'évaluation des performances du modèle."""

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Évalue le modèle sur les données de test."""
        # 1. Normaliser X_test (SANS refit !)
        X_test_scaled = self.preprocessor.transform(X_test)

        # 2. Prédire
        proba = self.model.predict(X_test_scaled)
        y_pred = (proba >= threshold).astype(int)

        # 3. Calculer les métriques
        print("\n=== RÉSULTATS DE L'ÉVALUATION ===")
        print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")

        print("\nMatrice de confusion :")
        print(confusion_matrix(y_test, y_pred))

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }