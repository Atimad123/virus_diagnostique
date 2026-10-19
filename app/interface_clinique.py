# app/interface_clinique.py

import joblib
from core.logistic_regression import LogisticRegressionModel
from core.predictor import ClinicalPredictor

def lancer_interface():
    print("=== BIENVENUE DANS L'INTERFACE CLINIQUE ===")
    
    # 1. Charger le modèle et le preprocessor
    modele = LogisticRegressionModel()
    modele.load("models/model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    # 2. Instancier le prédicteur (la logique métier)
    predictor = ClinicalPredictor(modele, preprocessor)

    # 3. Demander les infos du patient (exemple simple)
    try:
        age = int(input("Âge du patient : "))
        fever = int(input("Fièvre (1=oui, 0=non) : "))
        cough = int(input("Toux (1=oui, 0=non) : "))
        fatigue = int(input("Fatigue (1=oui, 0=non) : "))

        patient = {"age": age, "fever": fever, "cough": cough, "fatigue": fatigue}

        # 4. Appeler la classe métier
        proba = predictor.diagnose(patient)

        # 5. Prendre la décision (la condition est ici, dans l'app)
        if proba >= 0.5:
            diagnostic = "Infecté"
        else:
            diagnostic = "Sain"

        print(f"\nProbabilité d'infection : {proba:.2%}")
        print(f"Diagnostic final : {diagnostic}")

    except ValueError:
        print("Erreur : veuillez entrer des nombres valides.")