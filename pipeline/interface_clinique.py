# pipeline/interface_clinique.py

class ClinicalPredictor:
    def __init__(self, trained_model, preprocessor):
        self.model = trained_model
        self.preprocessor = preprocessor

    def diagnose(self, patient_data):
        patient_scaled = self.preprocessor.prepare_patient_data(patient_data)
        prediction = self.model.predict(patient_scaled)

        # Extraire le scalaire si c'est un array
        if hasattr(prediction, "__len__"):
            proba = float(prediction[0])
        else:
            proba = float(prediction)

        return proba