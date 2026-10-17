import pandas as pd
import numpy as np
import os

# Reproductibilité des résultats
np.random.seed(42)

# Nombre de patients fictifs
n = 500

# Génération des caractéristiques
age = np.random.randint(1, 90, n)
fever = np.random.randint(0, 2, n)
cough = np.random.randint(0, 2, n)
fatigue = np.random.randint(0, 2, n)

# Création d'une cible binaire synthétique
score = fever + cough + fatigue
probability = 1 / (1 + np.exp(-(score - 1.5)))

infected = np.random.binomial(1, probability)

# Création du DataFrame
df = pd.DataFrame({
    "age": age,
    "fever": fever,
    "cough": cough,
    "fatigue": fatigue,
    "infected": infected
})

# Création du dossier data s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Enregistrement du fichier CSV
df.to_csv("data/patient_data.csv", index=False)

print("Fichier patient_data.csv créé avec succès !")
print(df.head(10))
print("\nDimensions du dataset :", df.shape)
print("\nRépartition des classes :")
print(df["infected"].value_counts())