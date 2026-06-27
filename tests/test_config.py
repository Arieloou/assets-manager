"""Tests for features/config.py — configuration loading."""

import pytest
from features.config import (
    load_config,
    get_db_config,
    get_model_params,
    get_cookie_config,
    get_locations,
    get_hardware_states,
    get_equipment_types,
    get_risk_levels,
    get_vida_util_params,
    get_incidencias_lambda,
    get_costo_params,
)


class TestLoadConfig:
    """Verify config.yaml loads correctly and contains required sections."""

    def test_load_config_returns_dict(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_required_sections(self):
        # Verify that config.yaml contains all required non-sensitive application parameters
        config = load_config()
        required = ["app", "model_params", "equipment_types", "locations", "hardware_states", "risk_levels"]
        for section in required:
            assert section in config, f"Missing section: {section}"



class TestDbConfig:
    """Verify database configuration extraction."""

    def test_db_config_keys(self):
        db = get_db_config()
        for key in ["host", "port", "database", "user", "password"]:
            assert key in db, f"Missing key: {key}"

    def test_db_config_port_is_int(self):
        db = get_db_config()
        assert isinstance(db["port"], int)


class TestModelParams:
    """Verify model hyperparameter configuration."""

    def test_model_params_keys(self):
        params = get_model_params()
        for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "cv_folds"]:
            assert key in params, f"Missing param: {key}"

    def test_model_params_positive(self):
        params = get_model_params()
        assert params["n_estimators"] > 0
        assert params["cv_folds"] >= 2


class TestCookieConfig:
    def test_cookie_config_keys(self):
        cookie = get_cookie_config()
        for key in ["name", "key", "expiry_days"]:
            assert key in cookie


class TestListConfigs:
    """Verify list-type configurations."""

    def test_locations_non_empty(self):
        locations = get_locations()
        assert isinstance(locations, list)
        assert len(locations) > 0

    def test_hardware_states_non_empty(self):
        states = get_hardware_states()
        assert len(states) == 4

    def test_equipment_types_non_empty(self):
        types = get_equipment_types()
        assert len(types) > 0

    def test_risk_levels_non_empty(self):
        levels = get_risk_levels()
        assert len(levels) == 4

    def test_vida_util_params(self):
        params = get_vida_util_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_incidencias_lambda(self):
        lambdas = get_incidencias_lambda()
        assert isinstance(lambdas, dict)

    def test_costo_params(self):
        costos = get_costo_params()
        assert isinstance(costos, dict)
