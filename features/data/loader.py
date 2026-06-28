import csv

import pandas as pd
from features.database import get_all_equipos


class DataLoader:
    # Required columns for the dataset
    REQUIRED_COLUMNS = [
        'id_equipo',
        'vida_util_consumida',
        'tasa_incidencias_tecnicas',
        'tiempo_inactividad_acumulado',
        'costo_mto_reactivo_acumulado',
        'ubicacion_activo',
        'estado_integridad_hardware',
        'nivel_riesgo_operativo',
        'tipo_equipo'
    ]

    # Feature columns for model input (excludes ID_Equipo which is identifier)
    FEATURE_COLUMNS = [
        'vida_util_consumida',
        'tasa_incidencias_tecnicas',
        'tiempo_inactividad_acumulado',
        'costo_mto_reactivo_acumulado',
        'ubicacion_activo',
        'tipo_equipo'
    ]

    # Target columns
    TARGET_COLUMNS = [
        'estado_integridad_hardware',
        'nivel_riesgo_operativo'
    ]

    # Column mapping from database snake_case to spec PascalCase
    COLUMN_MAPPING = {
        'id_equipo': 'id_equipo',
        'vida_util_consumida': 'vida_util_consumida',
        'tasa_incidencias_tecnicas': 'tasa_incidencias_tecnicas',
        'tiempo_inactividad_acumulado': 'tiempo_inactividad_acumulado',
        'costo_mto_reactivo_acumulado': 'costo_mto_reactivo_acumulado',
        'ubicacion_activo': 'ubicacion_activo',
        'estado_integridad_hardware': 'estado_integridad_hardware',
        'nivel_riesgo_operativo': 'nivel_riesgo_operativo',
        'tipo_equipo': 'tipo_equipo',
    }

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file using pandas.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with normalized column names.
        """
        # Auto-detect delimiter (supports comma and semicolon CSV files)
        with open(filepath, 'r', encoding='utf-8') as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')

        df = pd.read_csv(filepath, sep=dialect.delimiter)

        return df

    @staticmethod
    def load_from_db() -> pd.DataFrame:
        """Load all equipos from PostgreSQL database.

        Returns:
            DataFrame with normalized column names.
        """
        df = get_all_equipos()

        # Rename columns from snake_case to PascalCase to match spec
        df = df.rename(columns=DataLoader.COLUMN_MAPPING)

        return df

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> bool:
        """Validate that dataframe has required columns.

        Args:
            df: DataFrame to validate.

        Returns:
            True if all required columns are present, False otherwise.
        """
        missing_columns = []
        for col in DataLoader.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_columns.append(col)

        if missing_columns:
            raise ValueError("Missing required columns: " + ", ".join(missing_columns))

        return True

    @staticmethod
    def get_features(df: pd.DataFrame) -> pd.DataFrame:
        """Return only the feature columns for model input.

        Args:
            df: DataFrame containing all columns.

        Returns:
            DataFrame with only feature columns.
        """
        # Validate that required columns exist
        DataLoader.validate_schema(df)

        # Return only feature columns that exist in the DataFrame
        available_features = [col for col in DataLoader.FEATURE_COLUMNS if col in df.columns]

        if not available_features:
            raise ValueError("No feature columns found in DataFrame")

        return df[available_features]

    @staticmethod
    def get_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Return target columns if present.

        Args:
            df: DataFrame containing all columns.

        Returns:
            DataFrame with target columns (Estado_Integridad_Hardware,
            Nivel_Riesgo_Operativo) if present.
        """
        available_targets = [col for col in DataLoader.TARGET_COLUMNS if col in df.columns]

        if not available_targets:
            # Return empty DataFrame with target column names if none present
            return pd.DataFrame(columns=DataLoader.TARGET_COLUMNS)

        return df[available_targets]
