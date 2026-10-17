from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Classe abstraite définissant l'interface commune à tous les modèles IA.
    
    Tout modèle (régression logistique, réseau de neurones, etc.) DOIT :
    - hériter de cette classe
    - implémenter les méthodes fit() et predict()
    
    Ainsi, ClinicalPredictor peut fonctionner avec n'importe quel modèle,
    tant qu'il possède une méthode predict(). C'est le POLYMORPHISME.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Entraîne le modèle sur les données X et les labels y.
        
        Args:
            X : features (DataFrame ou array numpy)
            y : cible (Series ou array numpy)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Prédit la probabilité d'infection pour chaque échantillon de X.
        
        Args:
            X : features (DataFrame ou array numpy)
        
        Returns:
            Un array numpy de probabilités entre 0 et 1.
            Exemple : [0.85, 0.12, 0.67]
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Sauvegarde le modèle entraîné sur le disque.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Charge un modèle entraîné depuis le disque.
        """
        pass