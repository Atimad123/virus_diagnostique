from core.dataset import DatasetLoader
from utils.preprocessing import Preprocessor

# 1. Charger les données
loader = DatasetLoader()
df = loader.load_data()
print("Données brutes :")
print(df.head())

# 2. Séparer train/test
X_train, X_test, y_train, y_test = loader.split_data()
print(f"\nX_train brut : {X_train.shape}")
print(f"X_test brut  : {X_test.shape}")

# 3. Prétraiter
preprocessor = Preprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled  = preprocessor.transform(X_test)

print(f"\nX_train normalisé : {X_train_scaled.shape}")
print(X_train_scaled.head())
print(f"\nMoyenne (doit être ~0) : {X_train_scaled.mean().round(2).to_dict()}")
print(f"Écart-type (doit être ~1) : {X_train_scaled.std().round(2).to_dict()}")

# 4. Tester prepare_patient_data
patient = {
    "age": 45, "temperature": 39.2, "fatigue": 1,
    "dry_cough": 1, "sore_throat": 0,
    "breathing_difficulty": 1, "headache": 1
}
patient_scaled = preprocessor.prepare_patient_data(patient)
print("\nPatient préparé :")
print(patient_scaled)