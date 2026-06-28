import pandas as pd
import numpy as np
from features.data.preprocessor import Preprocessor


class ModelPredictor:
    """Model predictor for hardware integrity and operational risk prediction."""
    
    def __init__(self, model, preprocessor):
        """Initialize the predictor with a trained model and preprocessor.
        
        Args:
            model: Trained multi-output model with predict method
            preprocessor: Preprocessor instance for feature preprocessing
        """
        self.model = model
        self.preprocessor = preprocessor
    
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
            'vida_util_consumida',
            'tasa_incidencias_tecnicas',
            'tiempo_inactividad_acumulado',
            'costo_mto_reactivo_acumulado',
            'ubicacion_activo_encoded',
            'tipo_equipo_encoded'
        ]
        return df[feature_cols]
    
    def predict(self, input_dict):
        """Predict single sample and return hardware integrity and risk level.
        
        Args:
            input_dict: Dictionary with keys:
                - vida_util_consumida (float)
                - tasa_incidencias_tecnicas (int)
                - tiempo_inactividad_acumulado (float)
                - costo_mto_reactivo_acumulado (float)
                - ubicacion_activo (string)
                - tipo_equipo (string)
                
        Returns:
            Tuple of (Estado_Integridad_Hardware, Nivel_Riesgo_Operativo)
        """
        X = self._prepare_single_sample(input_dict)
        
        predictions = self.model.predict(X)[0]
        estado_encoded = predictions[0]
        riesgo_encoded = predictions[1]
        
        estado_integridad = self.preprocessor.target_encoder.inverse_transform([[estado_encoded]])[0][0]
        riesgo = self.preprocessor.risk_encoder.inverse_transform([[riesgo_encoded]])[0][0]
        
        return (estado_integridad, riesgo)
    
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
                - estado_integridad_hardware
                - nivel_riesgo_operativo
        """
        df = df.copy()
        
        df = self.preprocessor.encode_categorical(df, fit=False)
        feature_cols = [
            'vida_util_consumida',
            'tasa_incidencias_tecnicas',
            'tiempo_inactividad_acumulado',
            'costo_mto_reactivo_acumulado',
            'ubicacion_activo_encoded',
            'tipo_equipo_encoded'
        ]
        X = df[feature_cols]
        
        predictions = self.model.predict(X)
        estados_encoded = predictions[:, 0]
        riesgos_encoded = predictions[:, 1]
        
        estados_integridad = self.preprocessor.target_encoder.inverse_transform(estados_encoded.reshape(-1, 1))
        riesgos = self.preprocessor.risk_encoder.inverse_transform(riesgos_encoded.reshape(-1, 1))
        
        df['estado_integridad_hardware'] = estados_integridad
        df['nivel_riesgo_operativo'] = riesgos
        
        return df
