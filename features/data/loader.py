import pandas as pd
from features.database import get_all_equipos


class DataLoader:
    # Required columns for the dataset
    REQUIRED_COLUMNS = [
        'ID_Equipo',
        'Vida_Util_Consumida',
        'Tasa_Incidencias_Tecnicas',
        'Tiempo_Inactividad_Acumulado',
        'Costo_Mto_Reactivo_Acumulado',
        'Ubicacion_Activo',
        'Estado_Integridad_Hardware',
        'Tipo_Equipo'
    ]

    # Feature columns for model input (excludes ID_Equipo which is identifier)
    FEATURE_COLUMNS = [
        'Vida_Util_Consumida',
        'Tasa_Incidencias_Tecnicas',
        'Tiempo_Inactividad_Acumulado',
        'Costo_Mto_Reactivo_Acumulado',
        'Ubicacion_Activo',
        'Tipo_Equipo'
    ]

    # Target columns
    TARGET_COLUMNS = [
        'Estado_Integridad_Hardware',
        'Nivel_Riesgo_Operativo'
    ]

    # Column mapping from database snake_case to spec PascalCase
    COLUMN_MAPPING = {
        'id_equipo': 'ID_Equipo',
        'vida_util_consumida': 'Vida_Util_Consumida',
        'tasa_incidencias_tecnicas': 'Tasa_Incidencias_Tecnicas',
        'tiempo_inactividad_acumulado': 'Tiempo_Inactividad_Acumulado',
        'costo_mto_reactivo_acumulado': 'Costo_Mto_Reactivo_Acumulado',
        'ubicacion_activo': 'Ubicacion_Activo',
        'estado_integridad_hardware': 'Estado_Integridad_Hardware',
        'tipo_equipo': 'Tipo_Equipo',
        'nivel_riesgo_operativo': 'Nivel_Riesgo_Operativo'
    }

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file using pandas.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with normalized column names.
        """
        df = pd.read_csv(filepath)

        # Rename columns to match spec PascalCase naming
        # Check if columns are in snake_case (database format) and rename them
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['id_equipo', 'id_equipo', 'idequipo', 'equipoid']:
                rename_dict[col] = 'ID_Equipo'
            elif col_lower == 'vida_util_consumida':
                rename_dict[col] = 'Vida_Util_Consumida'
            elif col_lower == 'tasa_incidencias_tecnicas':
                rename_dict[col] = 'Tasa_Incidencias_Tecnicas'
            elif col_lower == 'tiempo_inactividad_acumulado':
                rename_dict[col] = 'Tiempo_Inactividad_Acumulado'
            elif col_lower == 'costo_mto_reactivo_acumulado':
                rename_dict[col] = 'Costo_Mto_Reactivo_Acumulado'
            elif col_lower == 'ubicacion_activo':
                rename_dict[col] = 'Ubicacion_Activo'
            elif col_lower == 'estado_integridad_hardware':
                rename_dict[col] = 'Estado_Integridad_Hardware'
            elif col_lower == 'tipo_equipo':
                rename_dict[col] = 'Tipo_Equipo'
            elif col_lower == 'nivel_riesgo_operativo':
                rename_dict[col] = 'Nivel_Riesgo_Operativo'

        if rename_dict:
            df = df.rename(columns=rename_dict)

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
