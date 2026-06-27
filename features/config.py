import yaml
import streamlit as st
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

_config: Dict[str, Any] = None

def load_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config

def get_db_config() -> Dict[str, Any]:
    db_connection = st.secrets["connections"]["postgresql"]
    return {
        "host": db_connection["host"],
        "port": int(db_connection["port"]),
        "database": db_connection["database"],
        "user": db_connection["username"],
        "password": db_connection["password"],
    }

def get_cookie_config() -> Dict[str, Any]:
    cookie_config = st.secrets["secrets"]
    return {
        "name": cookie_config["cookie_name"],
        "key": cookie_config["cookie_key"],
        "expiry_days": int(cookie_config["cookie_expiry_days"]),
    }

def get_model_params() -> Dict[str, Any]:
    return load_config()["model_params"]

def get_vida_util_params() -> Dict[str, list]:
    return load_config()["vida_util_params"]

def get_incidencias_lambda() -> Dict[str, int]:
    return load_config()["incidencias_lambda"]

def get_costo_params() -> Dict[str, list]:
    return load_config()["costo_params"]

def get_locations() -> list:
    return load_config()["locations"]

def get_hardware_states() -> list:
    return load_config()["hardware_states"]

def get_risk_levels() -> list:
    return load_config()["risk_levels"]

def get_equipment_types() -> list:
    return load_config()["equipment_types"]
