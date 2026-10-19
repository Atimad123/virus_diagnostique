# debug_model.py
import joblib
import numpy as np

from core.logistic_regression import LogisticRegressionModel
from utils.preprocessing import Preprocessor

# Charger
modele = LogisticRegressionModel()
modele.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Voir les coefficients
print("=== Coefficients du modèle ===")
print("Features :", preprocessor.feature_columns_)
print("Coefficients :", modele.model.coef_)
print("Intercept :", modele.model.intercept_)

# Tester la normalisation d'un patient
patient = {"age": 30, "fever": 36.8, "cough": 0, "fatigue": 0}
scaled = preprocessor.prepare_patient_data(patient)
print("\n=== Patient sain normalisé ===")
print(scaled)

patient2 = {"age": 45, "fever": 39.2, "cough": 1, "fatigue": 1}
scaled2 = preprocessor.prepare_patient_data(patient2)
print("\n=== Patient infecté normalisé ===")
print(scaled2)

# Voir les moyennes apprises
print("\n=== Moyennes apprises (scaler.mean_) ===")
print(preprocessor.scaler.mean_)
print("=== Écarts-types appris (scaler.scale_) ===")
print(preprocessor.scaler.scale_)