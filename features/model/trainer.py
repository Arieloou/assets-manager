import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Any

from features.config import get_model_params


class ModelTrainer:
    """Trainer class for Random Forest model training.
    
    Trains a model to predict Estado_Integridad_Hardware (multi-class: Excelente, Bueno, Regular, Critico).
    """
    
    FEATURE_COLUMNS = [
        'vida_util_consumida',
        'tasa_incidencias_tecnicas',
        'tiempo_inactividad_acumulado',
        'costo_mto_reactivo_acumulado',
        'ubicacion_activo_encoded',
        'tipo_equipo_encoded'
    ]
    
    FEATURE_NAMES = [
        'vida_util_consumida',
        'tasa_incidencias_tecnicas',
        'tiempo_inactividad_acumulado',
        'costo_mto_reactivo_acumulado',
        'ubicacion_activo',
        'tipo_equipo'
    ]
    
    RISK_ENCODING_ORDER = ['Bajo', 'Medio', 'Alto', 'Critico']
    
    def __init__(self):
        """Initialize RandomForestClassifier with params from config."""
        # Load hyperparameters from config file
        params = get_model_params()
        
        # Setup the pipeline scaling numerical values and feeding to RandomForestClassifier
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=params.get('n_estimators', 50),
                criterion='entropy',
                class_weight='balanced',
                random_state=50
            ))
        ])
        self._feature_importances = None
    
    def train(self, X_train, y_train):
        """Fit the model and return self.
        
        Args:
            X_train: Training features (DataFrame or array)
            y_train: Training labels
            
        Returns:
            self
        """
        # Train the entire pipeline (StandardScaler and RandomForestClassifier)
        self.model.fit(X_train, y_train)
        
        # Retrieve the feature importances from the classifier step of the pipeline
        self._feature_importances = self.model.named_steps['classifier'].feature_importances_
        return self
    
    def train_risk(self, X_train, y_risk):
        """Fit a separate model for risk level prediction.
        
        Args:
            X_train: Training features (DataFrame or array)
            y_risk: Training risk labels
            
        Returns:
            self
        """
        self.model.fit(X_train, y_risk)
        self._feature_importances = self.model.named_steps['classifier'].feature_importances_
        return self
    
    def train_multioutput(self, X_train, y_multi):
        """Fit a single model for multi-output prediction (estado and riesgo).
        
        Args:
            X_train: Training features (DataFrame or array)
            y_multi: 2D array of targets (estado, riesgo) via np.column_stack
            
        Returns:
            self
        """
        params = get_model_params()
        rf_multi = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            criterion="entropy",
            random_state=50
        )
        rf_multi.fit(X_train, y_multi)
        self.model = rf_multi
        self._feature_importances = rf_multi.feature_importances_
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
        # Compute cross validation scores using the pipeline to prevent data leakage
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
            # Extract importance from the correct pipeline step if necessary
            if hasattr(self.model, 'feature_importances_'):
                self._feature_importances = self.model.feature_importances_
            else:
                self._feature_importances = self.model.named_steps['classifier'].feature_importances_
        
        return dict(zip(self.FEATURE_NAMES, self._feature_importances))
    
    def train_without_vida_util(self, X, y):
        """Train a variant excluding Vida_Util_Consumida column.
        
        Args:
            X: Full features DataFrame or array
            y: Labels
            
        Returns:
            self with a new model trained without Vida_Util_Consumida
        """
        # Load parameters from config file
        params = get_model_params()
        
        # Initialize a separate pipeline excluding the vida util column
        model_without_vida = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=params.get("n_estimators", 50),
                criterion="entropy",
                random_state=50
            ))
        ])
        
        vida_util_idx = self.FEATURE_COLUMNS.index("vida_util_consumida")
        
        # Extract features without the vida util column
        if hasattr(X, "columns"):
            X_without_vida = X.drop(columns=["vida_util_consumida"])
        else:
            X_without_vida = np.delete(X, vida_util_idx, axis=1)
        
        # Fit the model without the vida util feature
        model_without_vida.fit(X_without_vida, y)
        self.model = model_without_vida
        self._feature_importances = self.model.named_steps['classifier'].feature_importances_
        
        return self
    
    def save_model(self, filepath):
        """Save model using joblib.
        
        Args:
            filepath: Path where to save the model
        """
        # Save the entire pipeline object (including StandardScaler)
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load model using joblib.
        
        Args:
            filepath: Path to the saved model
        """
        # Load the saved pipeline object from disk
        self.model = joblib.load(filepath)
        
        # Get feature importances from pipeline step or estimator
        if hasattr(self.model, 'feature_importances_'):
            self._feature_importances = self.model.feature_importances_
        else:
            self._feature_importances = self.model.named_steps['classifier'].feature_importances_

