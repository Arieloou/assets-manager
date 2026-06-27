import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Any

from features.config import get_model_params


class ModelTrainer:
    """Trainer class for Random Forest model training.
    
    Trains a model to predict Estado_Integridad_Hardware (multi-class: Excelente, Bueno, Regular, Critico).
    """
    
    FEATURE_COLUMNS = [
        'Vida_Util_Consumida',
        'Tasa_Incidencias_Tecnicas',
        'Tiempo_Inactividad_Acumulado',
        'Costo_Mto_Reactivo_Acumulado',
        'Ubicacion_Activo_encoded',
        'Tipo_Equipo_encoded'
    ]
    
    FEATURE_NAMES = [
        'Vida_Util_Consumida',
        'Tasa_Incidencias_Tecnicas',
        'Tiempo_Inactividad_Acumulado',
        'Costo_Mto_Reactivo_Acumulado',
        'Ubicacion_Activo',
        'Tipo_Equipo'
    ]
    
    def __init__(self):
        """Initialize RandomForestClassifier with params from config."""
        params = get_model_params()
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42
        )
        self._feature_importances = None
    
    def train(self, X_train, y_train):
        """Fit the model and return self.
        
        Args:
            X_train: Training features (DataFrame or array)
            y_train: Training labels
            
        Returns:
            self
        """
        self.model.fit(X_train, y_train)
        self._feature_importances = self.model.feature_importances_
        return self
    
    def cross_validate(self, X, y, k=5):
        """Perform k-fold cross-validation.
        
        Args:
            X: Features for cross-validation
            y: Labels for cross-validation
            k: Number of folds (default 5)
            
        Returns:
            Dict with mean_accuracy, std_accuracy, and individual fold scores
        """
        scores = cross_val_score(self.model, X, y, cv=k)
        
        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": scores.tolist()
        }
    
    def get_feature_importance(self):
        """Return feature importances as dict.
        
        Returns:
            Dictionary mapping feature names to their importance values
        """
        if self._feature_importances is None:
            self._feature_importances = self.model.feature_importances_
        
        return dict(zip(self.FEATURE_NAMES, self._feature_importances))
    
    def train_without_vida_util(self, X, y):
        """Train a variant excluding Vida_Util_Consumida column.
        
        Args:
            X: Full features DataFrame or array
            y: Labels
            
        Returns:
            self with a new model trained without Vida_Util_Consumida
        """
        params = get_model_params()
        model_without_vida = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42
        )
        
        vida_util_idx = self.FEATURE_COLUMNS.index("Vida_Util_Consumida")
        
        if hasattr(X, "columns"):
            X_without_vida = X.drop(columns=["Vida_Util_Consumida"])
        else:
            X_without_vida = np.delete(X, vida_util_idx, axis=1)
        
        model_without_vida.fit(X_without_vida, y)
        self.model = model_without_vida
        self._feature_importances = self.model.feature_importances_
        
        return self
    
    def save_model(self, filepath):
        """Save model using joblib.
        
        Args:
            filepath: Path where to save the model
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load model using joblib.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self._feature_importances = self.model.feature_importances_
