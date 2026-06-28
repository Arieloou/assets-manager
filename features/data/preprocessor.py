import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class Preprocessor:
    """Data preprocessor for hardware integrity and operational risk prediction."""
    
    def __init__(self):
        self.encoders = {}
        self.target_encoder = None
        self.risk_encoder = None
        self.location_encoder = None
        self._feature_names = [
            'vida_util_consumida',
            'tasa_incidencias_tecnicas',
            'tiempo_inactividad_acumulado',
            'costo_mto_reactivo_acumulado',
            'ubicacion_activo',
            'tipo_equipo'
        ]
        self._target_names = ['estado_integridad_hardware', 'nivel_riesgo_operativo']
    
    def encode_categorical(self, df, fit=True):
        """Label encode categorical variables (Ubicacion_Activo, Tipo_Equipo).
        
        Args:
            df: DataFrame containing the categorical columns
            fit: If True, fit the encoders and transform. If False, only transform.
        
        Returns:
            DataFrame with encoded categorical columns
        """
        df = df.copy()
        
        categorical_columns = ['ubicacion_activo', 'tipo_equipo']
        
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
        """Encode Estado_Integridad_Hardware (target) using OrdinalEncoder.
        
        Args:
            df: DataFrame containing the target column
            fit: If True, fit the encoder and transform. If False, only transform.
        
        Returns:
            DataFrame with encoded target column
        """
        df = df.copy()
        
        orden_estado = [['Excelente', 'Bueno', 'Regular', 'Critico']]
        
        if fit:
            self.target_encoder = OrdinalEncoder(categories=orden_estado)
            df['estado_integridad_hardware_encoded'] = self.target_encoder.fit_transform(
                df[['estado_integridad_hardware']]
            )
        else:
            df['estado_integridad_hardware_encoded'] = self.target_encoder.transform(
                df[['estado_integridad_hardware']]
            )
        
        return df
    
    def encode_risk_target(self, df, fit=True):
        """Encode Nivel_Riesgo_Operativo (target) using OrdinalEncoder.
        
        Args:
            df: DataFrame containing the risk target column
            fit: If True, fit the encoder and transform. If False, only transform.
        
        Returns:
            DataFrame with encoded risk target column
        """
        df = df.copy()
        
        orden_riesgo = [['Bajo', 'Medio', 'Alto', 'Critico']]
        
        if fit:
            self.risk_encoder = OrdinalEncoder(categories=orden_riesgo)
            df['nivel_riesgo_operativo_encoded'] = self.risk_encoder.fit_transform(
                df[['nivel_riesgo_operativo']]
            )
        else:
            df['nivel_riesgo_operativo_encoded'] = self.risk_encoder.transform(
                df[['nivel_riesgo_operativo']]
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
            'vida_util_consumida',
            'tasa_incidencias_tecnicas',
            'tiempo_inactividad_acumulado',
            'costo_mto_reactivo_acumulado',
            'ubicacion_activo_encoded',
            'tipo_equipo_encoded'
        ]
        
        target_cols = [
            'estado_integridad_hardware_encoded',
            'nivel_riesgo_operativo_encoded',
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
