import pandas as pd
import numpy as np
from features.data.preprocessor import Preprocessor


class ModelPredictor:
    """Model predictor for hardware integrity and operational risk prediction."""
    
    # Mapping from Estado_Integridad_Hardware to Nivel_Riesgo_Operativo
    RISK_MAPPING = {
        'Excelente': 'Bajo',
        'Bueno': 'Medio',
        'Regular': 'Alto',
        'Critico': 'Critico'
    }
    
    def __init__(self, model, preprocessor):
        """Initialize the predictor with a trained model and preprocessor.
        
        Args:
            model: Trained model with predict and predict_proba methods
            preprocessor: Preprocessor instance for feature preprocessing
        """
        self.model = model
        self.preprocessor = preprocessor
    
    def _derive_risk_level(self, estado_integridad):
        """Derive Nivel_Riesgo_Operativo from Estado_Integridad_Hardware.
        
        Args:
            estado_integridad: Predicted Estado_Integridad_Hardware label
            
        Returns:
            Corresponding Nivel_Riesgo_Operativo label
        """
        return self.RISK_MAPPING.get(estado_integridad, estado_integridad)
    
    def _prepare_single_sample(self, input_dict):
        """Convert input dict to a DataFrame ready for prediction.
        
        Args:
            input_dict: Dictionary with feature values
            
        Returns:
            DataFrame with preprocessed features
        """
        df = pd.DataFrame([input_dict])
        df = self.preprocessor.encode_categorical(df, fit=False)
        feature_cols = [
            'Vida_Util_Consumida',
            'Tasa_Incidencias_Tecnicas',
            'Tiempo_Inactividad_Acumulado',
            'Costo_Mto_Reactivo_Acumulado',
            'Ubicacion_Activo_encoded',
            'Tipo_Equipo_encoded'
        ]
        return df[feature_cols]
    
    def predict(self, input_dict):
        """Predict single sample and return hardware integrity and risk level.
        
        Args:
            input_dict: Dictionary with keys:
                - Vida_Util_Consumida (float)
                - Tasa_Incidencias_Tecnicas (int)
                - Tiempo_Inactividad_Acumulado (float)
                - Costo_Mto_Reactivo_Acumulado (float)
                - Ubicacion_Activo (string)
                - Tipo_Equipo (string)
                
        Returns:
            Tuple of (Estado_Integridad_Hardware, Nivel_Riesgo_Operativo)
        """
        X = self._prepare_single_sample(input_dict)
        
        estado_encoded = self.model.predict(X)[0]
        estado_integridad = self.preprocessor.target_encoder.inverse_transform([estado_encoded])[0]
        
        nivel_riesgo = self._derive_risk_level(estado_integridad)
        
        return (estado_integridad, nivel_riesgo)
    
    def predict_proba(self, input_dict):
        """Return prediction probabilities for each class.
        
        Args:
            input_dict: Dictionary with feature values
            
        Returns:
            Array of prediction probabilities for each class
        """
        X = self._prepare_single_sample(input_dict)
        return self.model.predict_proba(X)[0]
    
    def predict_batch(self, df):
        """Predict on a DataFrame and return DataFrame with predictions added.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with original data plus predictions:
                - Estado_Integridad_Hardware
                - Nivel_Riesgo_Operativo
        """
        df = df.copy()
        
        df = self.preprocessor.encode_categorical(df, fit=False)
        feature_cols = [
            'Vida_Util_Consumida',
            'Tasa_Incidencias_Tecnicas',
            'Tiempo_Inactividad_Acumulado',
            'Costo_Mto_Reactivo_Acumulado',
            'Ubicacion_Activo_encoded',
            'Tipo_Equipo_encoded'
        ]
        X = df[feature_cols]
        
        estados_encoded = self.model.predict(X)
        estados_integridad = self.preprocessor.target_encoder.inverse_transform(estados_encoded)
        
        niveles_riesgo = [self._derive_risk_level(e) for e in estados_integridad]
        
        df['Estado_Integridad_Hardware'] = estados_integridad
        df['Nivel_Riesgo_Operativo'] = niveles_riesgo
        
        return df
