# Virus Diag 🦠

Système de diagnostic médical basé sur l'IA pour détecter l'infection virale chez un patient.

## Architecture

- `core/` : logique métier (dataset, modèles)
- `utils/` : outils (prétraitement)
- `pipeline/` : orchestration (entraînement, prédiction)
- `app/` : API et interface
- `data/` : données patients

## Installation

```bash
pip install pandas scikit-learn