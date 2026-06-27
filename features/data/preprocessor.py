import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    """Data preprocessor for hardware integrity and operational risk prediction."""
    
    def __init__(self):
        self.encoders = {}
        self.target_encoder = None
        self.risk_encoder = None
        self._feature_names = [
            'Vida_Util_Consumida',
            'Tasa_Incidencias_Tecnicas',
            'Tiempo_Inactividad_Acumulado',
            'Costo_Mto_Reactivo_Acumulado',
            'Ubicacion_Activo',
            'Tipo_Equipo'
        ]
        self._target_names = ['Estado_Integridad_Hardware', 'Nivel_Riesgo_Operativo']
    
    def encode_categorical(self, df, fit=True):
        """Label encode categorical variables (Ubicacion_Activo, Tipo_Equipo).
        
        Args:
            df: DataFrame containing the categorical columns
            fit: If True, fit the encoders and transform. If False, only transform.
        
        Returns:
            DataFrame with encoded categorical columns
        """
        df = df.copy()
        
        categorical_columns = ['Ubicacion_Activo', 'Tipo_Equipo']
        
        for col in categorical_columns:
            if fit:
                encoder = LabelEncoder()
                df[col + '_encoded'] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
            else:
                encoder = self.encoders[col]
                df[col + '_encoded'] = encoder.transform(df[col].astype(str))
        
        return df
    
    def encode_target(self, df, fit=True):
        """Encode Estado_Integridad_Hardware (target) using LabelEncoder.
        
        Args:
            df: DataFrame containing the target column
            fit: If True, fit the encoder and transform. If False, only transform.
        
        Returns:
            DataFrame with encoded target column
        """
        df = df.copy()
        
        if fit:
            self.target_encoder = LabelEncoder()
            df['Estado_Integridad_Hardware_encoded'] = self.target_encoder.fit_transform(
                df['Estado_Integridad_Hardware'].astype(str)
            )
        else:
            df['Estado_Integridad_Hardware_encoded'] = self.target_encoder.transform(
                df['Estado_Integridad_Hardware'].astype(str)
            )
        
        return df
    
    def encode_risk_level(self, series):
        """Encode Nivel_Riesgo_Operativo (Bajo, Medio, Alto, Critico).
        
        Args:
            series: Pandas Series containing risk level values
        
        Returns:
            Numpy array with encoded risk levels
        """
        risk_mapping = {
            'Bajo': 0,
            'Medio': 1,
            'Alto': 2,
            'Critico': 3
        }
        
        return series.map(risk_mapping)
    
    def split_features_targets(self, df):
        """Separate feature columns from target columns.
        
        Args:
            df: DataFrame with all columns
        
        Returns:
            Tuple of (features DataFrame, targets DataFrame)
        """
        feature_cols = [
            'Vida_Util_Consumida',
            'Tasa_Incidencias_Tecnicas',
            'Tiempo_Inactividad_Acumulado',
            'Costo_Mto_Reactivo_Acumulado',
            'Ubicacion_Activo_encoded',
            'Tipo_Equipo_encoded'
        ]
        
        target_cols = [
            'Estado_Integridad_Hardware_encoded',
            'Nivel_Riesgo_Operativo_encoded'
        ]
        
        features = df[feature_cols].copy()
        targets = df[target_cols].copy()
        
        return features, targets
    
    def get_feature_names(self):
        """Return list of feature column names after encoding.
        
        Returns:
            List of feature column names
        """
        return self._feature_names
