# pipeline/interface_clinique.py

class ClinicalPredictor:
    """
    Reçoit un modèle entraîné ET un preprocessor (pour normaliser les nouvelles données).
    Retourne la PROBABILITÉ brute. La décision (seuil) est prise dans main.py.
    """

    def __init__(self, trained_model, preprocessor):
        self.model = trained_model
        self.preprocessor = preprocessor

    def diagnose(self, patient_data):
        """
        Prend un dictionnaire patient, le normalise,
        et retourne la probabilité d'infection (entre 0 et 1).
        """
        # 1. Préparer les données du patient (normalisation identique à l'entraînement)
        patient_scaled = self.preprocessor.prepare_patient_data(patient_data)

        # 2. Prédire la probabilité
        prediction = self.model.predict(patient_scaled)
        proba = prediction[0] if hasattr(prediction, '__len__') else prediction

        return proba