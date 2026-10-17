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

    # Charger le modèle et le preprocessor sauvegardés
    from core.logistic_regression import LogisticRegressionModel

    modele_charge = LogisticRegressionModel()
    modele_charge.load("models/model.pkl")

    preprocessor_charge = joblib.load("models/preprocessor.pkl")

    # Créer le prédicteur clinique
    predictor = ClinicalPredictor(modele_charge, preprocessor_charge)

    # Nouveau patient (symptômes saisis par un médecin)
    patient = {
        "age": 45,
        "temperature": 39.2,
        "fatigue": 1,
        "dry_cough": 1,
        "sore_throat": 0,
        "breathing_difficulty": 1,
        "headache": 1,
    }

    # Prédire la probabilité
    proba = predictor.diagnose(patient)

    # LA CONDITION EST ICI, dans main.py
    if proba >= 0.5:
        diagnostic = "Infecté"
    else:
        diagnostic = "Sain"

    print(f"\nPatient : {patient}")
    print(f"Probabilité d'infection : {proba:.2%}")
    print(f"Diagnostic : {diagnostic}")