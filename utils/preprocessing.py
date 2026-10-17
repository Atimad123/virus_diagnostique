import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    """
    Classe responsable du nettoyage et de la préparation des données.
    Elle est conçue pour être utilisée dans le pipeline d'entraînement.
    """

    def __init__(self):
        # On initialise les outils de scikit-learn
        # SimpleImputer : remplace les valeurs manquantes par la médiane
        self.imputer = SimpleImputer(strategy="median")
        # StandardScaler : centre et réduit les données (moyenne=0, écart-type=1)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def clean_data(self, df):
        """
        Nettoie le DataFrame brut :
        - Supprime les colonnes inutiles (patient_id)
        - Gère les valeurs manquantes
        """
        df = df.copy()

        # 1. Supprimer la colonne patient_id si elle existe
        # Elle n'a aucune valeur prédictive pour le modèle
        if "patient_id" in df.columns:
            df = df.drop(columns=["patient_id"])

        # 2. Vérifier qu'il ne reste pas de colonnes non numériques
        # (au cas où il y aurait des colonnes texte inattendues)
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Attention : colonnes non numériques détectées et supprimées : {list(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)

        return df

    def fit(self, X_train):
        """
        Apprend les paramètres de normalisation sur les données d'entraînement.
        (Ne jamais fit sur les données de test !)
        """
        # On apprend la médiane pour l'imputation
        self.imputer.fit(X_train)
        # On apprend la moyenne et l'écart-type pour la normalisation
        X_imputed = self.imputer.transform(X_train)
        self.scaler.fit(X_imputed)

        self.is_fitted = True
        print("Preprocessor : fit terminé (paramètres appris sur X_train).")
        return self

    def transform(self, X):
        """
        Applique le nettoyage et la normalisation à un jeu de données.
        Utilisable sur X_train, X_test, ou sur une nouvelle donnée patient.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Le Preprocessor n'est pas encore entraîné. "
                "Appelez d'abord fit(X_train)."
            )

        # 1. Imputation des valeurs manquantes
        X_imputed = self.imputer.transform(X)

        # 2. Normalisation
        X_scaled = self.scaler.transform(X_imputed)

        # On retourne un DataFrame pour garder les noms de colonnes
        # (pratique pour le débogage et l'interface clinique)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def fit_transform(self, X_train):
        """
        Méthode raccourcie : fit puis transform sur les données d'entraînement.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def prepare_patient_data(self, patient_dict):
        """
        Méthode spéciale pour l'interface clinique.
        Prend un dictionnaire {feature: valeur} d'un seul patient
        et retourne un tableau normalisé prêt pour la prédiction.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Le Preprocessor doit être entraîné avant de préparer un patient."
            )

        # Convertir le dictionnaire en DataFrame d'une seule ligne
        df_patient = pd.DataFrame([patient_dict])

        # Nettoyage (suppression patient_id si présent)
        df_patient = self.clean_data(df_patient)

        # Vérification : les colonnes doivent correspondre à celles vues à l'entraînement
        expected_cols = list(self.scaler.feature_names_in_)
        missing = set(expected_cols) - set(df_patient.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes pour le patient : {missing}")

        # Réordonner les colonnes dans le même ordre qu'à l'entraînement
        df_patient = df_patient[expected_cols]

        # Appliquer la même transformation
        return self.transform(df_patient)