# core/predictor.py

class ClinicalPredictor:
    """
    Classe métier responsable de la prédiction clinique.
    Elle reçoit un modèle et un preprocessor, et retourne une probabilité.
    """
    def __init__(self, trained_model, preprocessor):
        self.model = trained_model
        self.preprocessor = preprocessor

    def diagnose(self, patient_data):
        # 1. Prétraiter les données du patient
        patient_scaled = self.preprocessor.prepare_patient_data(patient_data)
        
        # 2. Obtenir la prédiction du modèle
        prediction = self.model.predict(patient_scaled)
        proba = prediction[0] if hasattr(prediction, '__len__') else prediction
        
        # 3. Retourner la probabilité
        return proba