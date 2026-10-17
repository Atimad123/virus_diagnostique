# pipeline/interface_clinique.py

class ClinicalPredictor:
    def __init__(self, trained_model):
        self.model = trained_model

    def diagnose(self, patient_data):
        """Retourne UNIQUEMENT la probabilité brute."""
        prediction = self.model.predict(patient_data)
        proba = prediction[0] if hasattr(prediction, '__len__') else prediction
        return proba