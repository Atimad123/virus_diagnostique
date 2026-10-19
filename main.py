# main.py

import joblib
from core.dataset import DatasetLoader
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from core.predictor import ClinicalPredictor  # <-- IMPORT CORRIGÉ ICI
from core.logistic_regression import LogisticRegressionModel


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

    modele_charge = LogisticRegressionModel()
    modele_charge.load("models/model.pkl")
    preprocessor_charge = joblib.load("models/preprocessor.pkl")

    # Instanciation du prédicteur depuis core
    predictor = ClinicalPredictor(modele_charge, preprocessor_charge)

    # Patient avec fièvre + symptômes (fever est BINAIRE : 1=fièvre, 0=pas de fièvre)
    patient_malade = {
        "age": 45,
        "fever": 1,        # a de la fièvre
        "cough": 1,        # tousse
        "fatigue": 1,      # est fatigué
    }

    # Patient sain (pas de fièvre, pas de symptômes)
    patient_sain = {
        "age": 30,
        "fever": 0,        # pas de fièvre
        "cough": 0,        # ne tousse pas
        "fatigue": 0,      # pas fatigué
    }

    for label, patient in [("Patient malade", patient_malade),
                            ("Patient sain", patient_sain)]:
        proba = predictor.diagnose(patient)
        # La condition métier est bien dans main.py, comme demandé par votre prof
        diagnostic = "Infecté" if proba >= 0.5 else "Sain"

        print(f"\n{label} : {patient}")
        print(f"Probabilité d'infection : {proba:.2%}")
        print(f"Diagnostic : {diagnostic}")