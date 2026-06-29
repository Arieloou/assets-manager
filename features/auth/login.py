import streamlit as st
import streamlit_authenticator as stauth
import yaml
from pathlib import Path


from features.config import get_cookie_config


def to_dict(value):
    # Recursively convert dict-like/AttrDict objects to standard mutable dicts
    if hasattr(value, "keys"):
        return {k: to_dict(value[k]) for k in value.keys()}
    elif isinstance(value, list):
        return [to_dict(v) for v in value]
    return value


def load_credentials() -> dict:
    # Attempt to load credentials from Streamlit secrets config
    try:
        if "credentials" in st.secrets:
            raw = st.secrets["credentials"]
            # Convert nested AttrDict structures to standard mutable dicts
            if hasattr(raw, "keys"):
                raw_dict = to_dict(raw)
                if "usernames" in raw_dict:
                    return {"usernames": raw_dict["usernames"]}
                return {"usernames": raw_dict}
    except Exception:
        # Ignore secret loading issues and proceed to local fallback
        pass

    # Load local credentials file if secrets config is not set up
    credentials_path = Path(__file__).parent / "credentials.yaml"
    if credentials_path.exists():
        with open(credentials_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw:
            return to_dict(raw.get("credentials", raw))

    # Return empty dict if no credentials source is available
    return {"usernames": {}}

def is_pre_hashed() -> bool:
    try:
        if "credentials" in st.secrets:
            return True
    except Exception:
        pass
    return False


def initialize_authenticator() -> stauth.Authenticate:
    credentials = load_credentials()
    cookie_config = get_cookie_config()

    auto_hash = not is_pre_hashed()

    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name=cookie_config["name"],
        cookie_key=cookie_config["key"],
        cookie_expiry_days=cookie_config["expiry_days"],
        auto_hash=auto_hash
    )

    return authenticator


def authenticate_user():
    authenticator = initialize_authenticator()
    return authenticator


def require_auth():
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    if not st.session_state.get("authentication_status"):
        st.warning("Por favor, inicia sesion para acceder a esta pagina.")
        return False

    return True
