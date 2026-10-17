# main.py

import joblib
from pathlib import Path

from core.dataset import DatasetLoader
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.interface_clinique import ClinicalPredictor


if __name__ == "__main__":

    # =====================================================
    # PARTIE 1 : Chargement et séparation des données
    # =====================================================
    print("=" * 60)
    print("ÉTAPE 1 : Chargement des données")
    print("=" * 60)

    loader = DatasetLoader()
    X_train, X_test, y_train, y_test = loader.split_data()

    # =====================================================
    # PARTIE 2 : Entraînement
    # =====================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Entraînement du modèle")
    print("=" * 60)

    trainer = Trainer()
    trainer.train(X_train, y_train)
    trainer.save()

    # =====================================================
    # PARTIE 3 : Évaluation
    # =====================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Évaluation")
    print("=" * 60)

    evaluator = Evaluator(trainer.model, trainer.preprocessor)
    evaluator.evaluate(X_test, y_test)
    # =====================================================
    # PARTIE 4 : Prédiction clinique pour un patient
    # =====================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Diagnostic d'un nouveau patient")
    print("=" * 60)

    from core.logistic_regression import LogisticRegressionModel
    import joblib

    # Charger le modèle et le preprocessor sauvegardés
    modele_charge = LogisticRegressionModel()
    modele_charge.load("models/model.pkl")
    preprocessor_charge = joblib.load("models/preprocessor.pkl")

    predictor = ClinicalPredictor(modele_charge, preprocessor_charge)

    # Patient : les clés DOIVENT correspondre exactement aux colonnes du CSV
    patient = {
        "age": 45,
        "fever": 39.2,
        "cough": 1,
        "fatigue": 1,
    }

    # Prédire la probabilité
    proba = predictor.diagnose(patient)

    # La condition est dans main.py (selon la consigne de la prof)
    if proba >= 0.5:
        diagnostic = "Infecté"
    else:
        diagnostic = "Sain"

    print(f"\nPatient : {patient}")
    print(f"Probabilité d'infection : {proba:.2%}")
    print(f"Diagnostic : {diagnostic}")

    # --- Bonus : tester un patient sain pour comparaison ---
    patient_sain = {
        "age": 30,
        "fever": 36.8,
        "cough": 0,
        "fatigue": 0,
    }

    proba_sain = predictor.diagnose(patient_sain)
    diagnostic_sain = "Infecté" if proba_sain >= 0.5 else "Sain"

    print(f"\nPatient sain : {patient_sain}")
    print(f"Probabilité d'infection : {proba_sain:.2%}")
    print(f"Diagnostic : {diagnostic_sain}")