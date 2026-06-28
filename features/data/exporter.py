import csv
import io
from datetime import datetime

import pandas as pd
import streamlit as st

from features.database import save_equipo, get_all_equipos, get_session, HistoricalData


def get_risk_level(estado_integridad: str) -> str:
    """Calculate risk level based on hardware integrity state."""
    risk_mapping = {
        "Excelente": "Bajo",
        "Bueno": "Medio",
        "Regular": "Alto",
        "Critico": "Critico",
        "Crítico": "Critico"
    }
    return risk_mapping.get(estado_integridad, "Medio")


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has required columns."""
    required_columns = [
        "id_equipo",
        "vida_util_consumida",
        "tasa_incidencias_tecnicas",
        "tiempo_inactividad_acumulado",
        "costo_mto_reactivo_acumulado",
        "ubicacion_activo",
        "estado_integridad_hardware",
        "tipo_equipo"
    ]

    return all(col in df.columns for col in required_columns)


def import_csv(uploaded_file) -> pd.DataFrame:
    """Import a CSV file and return a DataFrame with added timestamp and risk level.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        DataFrame with added timestamp_registro and nivel_riesgo_operativo columns
        
    Raises:
        ValueError: If CSV schema is invalid
    """
    # Auto-detect delimiter (supports comma and semicolon CSV files)
    sample = uploaded_file.read(4096)
    if isinstance(sample, bytes):
        sample = sample.decode("utf-8")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","
    uploaded_file.seek(0)
    
    df = pd.read_csv(uploaded_file, sep=delimiter)
    
    if not validate_schema(df):
        raise ValueError("Invalid CSV schema. Required columns: " + ", ".join([
            "id_equipo",
            "vida_util_consumida",
            "tasa_incidencias_tecnicas",
            "tiempo_inactividad_acumulado",
            "costo_mto_reactivo_acumulado",
            "ubicacion_activo",
            "estado_integridad_hardware",
            "tipo_equipo"
        ]))
    
    # Calculate risk level and add timestamp column
    df["nivel_riesgo_operativo"] = df["estado_integridad_hardware"].apply(get_risk_level)
    df["timestamp_registro"] = datetime.now()
        
    return df


def export_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV bytes ready for download
    """
    return df.to_csv(index=False).encode("utf-8")


def save_to_database(df: pd.DataFrame) -> int:
    """Save DataFrame records to PostgreSQL database.
    
    Args:
        df: DataFrame containing equipment records
        
    Returns:
        Number of records saved
    """
    saved_count = 0
    
    for _, row in df.iterrows():
        equipo_data = {
            "id_equipo": row["id_equipo"],
            "vida_util_consumida": row["vida_util_consumida"],
            "tasa_incidencias_tecnicas": row["tasa_incidencias_tecnicas"],
            "tiempo_inactividad_acumulado": row["tiempo_inactividad_acumulado"],
            "costo_mto_reactivo_acumulado": row["costo_mto_reactivo_acumulado"],
            "ubicacion_activo": row["ubicacion_activo"],
            "estado_integridad_hardware": row["estado_integridad_hardware"],
            "tipo_equipo": row["tipo_equipo"],
            "nivel_riesgo_operativo": row["nivel_riesgo_operativo"],
            "timestamp_registro": row["timestamp_registro"],
        }
        save_equipo(equipo_data)
        saved_count += 1
    
    return saved_count


def get_historical_data() -> pd.DataFrame:
    """Retrieve all historical data from the database.
    
    Returns:
        DataFrame containing all historical equipment data
    """
    return get_all_equipos()
