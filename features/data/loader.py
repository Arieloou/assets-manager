import csv

import pandas as pd
from features.database import get_all_devices


class DataLoader:
    """Loads the device dataset and exposes raw feature / target columns.

    Feature engineering (date differences, encoders, scaling) is delegated to
    ``features.data.preprocessor.Preprocessor``.
    """

    # Raw columns expected in the dataset / database
    REQUIRED_COLUMNS = [
        'device_id',
        'device_brand',
        'device_type',
        'acquisition_date',
        'technical_incident_rate',
        'last_reactive_maintenance_date',
        'last_preventive_maintenance_date',
        'headquarters_location',
        'hardware_integrity_status',
        'operational_risk_level',
    ]

    # Raw columns used to derive the model features (excludes device_id, which is
    # an identifier, and the target column).
    FEATURE_COLUMNS = [
        'device_brand',
        'device_type',
        'hardware_integrity_status',
        'headquarters_location',
        'acquisition_date',
        'technical_incident_rate',
        'last_reactive_maintenance_date',
        'last_preventive_maintenance_date',
    ]

    # Single target column
    TARGET_COLUMNS = [
        'operational_risk_level',
    ]

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load the dataset from a CSV file.

        Auto-detects the delimiter (the provided dataset uses ``;``) and reads
        with ``utf-8-sig`` to strip the BOM.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with the raw dataset columns.
        """
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ';'

        return pd.read_csv(filepath, sep=delimiter, encoding='utf-8-sig')

    @staticmethod
    def load_from_db() -> pd.DataFrame:
        """Load all devices from the database."""
        return get_all_devices()

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has the required columns.

        ``operational_risk_level`` is optional here so that prediction-only
        DataFrames (without the target) still pass.
        """
        required = [c for c in DataLoader.REQUIRED_COLUMNS if c != 'operational_risk_level']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        return True

    @staticmethod
    def get_features(df: pd.DataFrame) -> pd.DataFrame:
        """Return only the raw feature columns used by the preprocessor."""
        DataLoader.validate_schema(df)

        available_features = [col for col in DataLoader.FEATURE_COLUMNS if col in df.columns]
        if not available_features:
            raise ValueError("No feature columns found in DataFrame")

        return df[available_features].copy()

    @staticmethod
    def get_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Return the target column (operational_risk_level) if present."""
        available_targets = [col for col in DataLoader.TARGET_COLUMNS if col in df.columns]
        if not available_targets:
            return pd.DataFrame(columns=DataLoader.TARGET_COLUMNS)
        return df[available_targets].copy()
