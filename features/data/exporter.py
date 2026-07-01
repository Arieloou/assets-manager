import csv
from datetime import datetime

import pandas as pd

from features.database import save_device, get_all_devices


REQUIRED_COLUMNS = [
    "device_id",
    "device_brand",
    "device_type",
    "acquisition_date",
    "technical_incident_rate",
    "last_reactive_maintenance_date",
    "last_preventive_maintenance_date",
    "headquarters_location",
    "hardware_integrity_status",
    "operational_risk_level",
]


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has the required dataset columns.

    ``last_reactive_maintenance_date`` may be entirely absent only if the column
    is missing from the file; null values within it are allowed.
    """
    return all(col in df.columns for col in REQUIRED_COLUMNS)


def _parse_date(value):
    """Parse a single date value (day-first); return None when missing/invalid."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    ts = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def import_csv(uploaded_file) -> pd.DataFrame:
    """Import a CSV file and return a DataFrame with an added timestamp.

    The dataset already carries the ``operational_risk_level`` target, so no
    risk derivation is performed here.

    Raises:
        ValueError: If the CSV schema is invalid.
    """
    sample = uploaded_file.read(4096)
    if isinstance(sample, bytes):
        sample = sample.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"
    uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file, sep=delimiter, encoding="utf-8-sig")

    if not validate_schema(df):
        raise ValueError(
            "Invalid CSV schema. Required columns: " + ", ".join(REQUIRED_COLUMNS)
        )

    df["registered_at"] = datetime.now()
    return df

def save_to_database(df: pd.DataFrame, progress_callback=None) -> int:
    """Persist DataFrame records to the database. Returns the count saved.

    Args:
        df: DataFrame with the dataset columns.
        progress_callback: Optional callable invoked as ``callback(done, total)``
            after each saved record, so callers can render progress.
    """
    saved_count = 0
    total = len(df)

    for _, row in df.iterrows():
        device_data = {
            "device_id": str(row["device_id"]),
            "device_brand": row.get("device_brand"),
            "acquisition_date": _parse_date(row.get("acquisition_date")),
            "technical_incident_rate": int(row["technical_incident_rate"]),
            "last_reactive_maintenance_date": _parse_date(row.get("last_reactive_maintenance_date")),
            "last_preventive_maintenance_date": _parse_date(row.get("last_preventive_maintenance_date")),
            "headquarters_location": row["headquarters_location"],
            "hardware_integrity_status": row["hardware_integrity_status"],
            "device_type": row["device_type"],
            "operational_risk_level": row.get("operational_risk_level"),
            "registered_at": row.get("registered_at", datetime.now()),
        }
        save_device(device_data)
        saved_count += 1
        if progress_callback is not None:
            progress_callback(saved_count, total)

    return saved_count


def get_historical_data() -> pd.DataFrame:
    """Retrieve all historical equipment data from the database."""
    return get_all_devices()
