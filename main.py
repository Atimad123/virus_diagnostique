# main.py
from pipeline.interface_clinique import ClinicalPredictor

if __name__ == "__main__":
    class FakeModel:
        def predict(self, data):
            return [0.85]

    predictor = ClinicalPredictor(FakeModel())
    patient = {"age": 45, "temperature": 39}

    # La probabilité est calculée par le modèle
    proba = predictor.diagnose(patient)

    if proba >= 0.5:
        diagnostic = "Infecté"
    else:
        diagnostic = "Sain"

    print(f"Probabilité : {proba}")
    print(f"Diagnostic : {diagnostic}")