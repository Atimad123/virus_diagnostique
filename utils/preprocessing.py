import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    """
    Classe responsable du nettoyage et de la préparation des données.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False
        # On stocke nous-mêmes les colonnes vues à l'entraînement
        self.feature_columns_ = None

    def clean_data(self, df):
        """Nettoie le DataFrame brut."""
        df = df.copy()

        if "patient_id" in df.columns:
            df = df.drop(columns=["patient_id"])

        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Attention : colonnes non numériques supprimées : {list(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)

        return df

    def fit(self, X_train):
        """Apprend les paramètres sur X_train."""
        # 1. Sauvegarder les noms de colonnes vus à l'entraînement
        self.feature_columns_ = list(X_train.columns)

        # 2. Apprendre l'imputation
        self.imputer.fit(X_train)
        X_imputed = self.imputer.transform(X_train)

        # 3. Apprendre la normalisation
        self.scaler.fit(X_imputed)

        self.is_fitted = True
        print(f"Preprocessor : fit terminé sur {len(self.feature_columns_)} colonnes.")
        print(f"Colonnes apprises : {self.feature_columns_}")
        return self

    def transform(self, X):
        """Applique la transformation à un jeu de données."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor non entraîné. Appelez fit() d'abord.")

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=X.columns)

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

    def prepare_patient_data(self, patient_dict):
        """
        Prépare UN patient (dict) pour la prédiction.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor non entraîné.")

        # 1. Convertir le dict en DataFrame
        df_patient = pd.DataFrame([patient_dict])

        # 2. Vérifier que toutes les colonnes attendues sont présentes
        missing = set(self.feature_columns_) - set(df_patient.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour le patient : {missing}\n"
                f"Colonnes attendues : {self.feature_columns_}"
            )

        # 3. Réordonner les colonnes dans le même ordre qu'à l'entraînement
        df_patient = df_patient[self.feature_columns_]

        # 4. Appliquer la même transformation
        return self.transform(df_patient)