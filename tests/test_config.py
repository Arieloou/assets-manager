"""Tests for features/config.py — configuration loading."""

import pandas as pd
import pytest

from features.config import (
    load_config,
    get_model_params,
    get_locations,
    get_brands,
    get_hardware_states,
    get_device_types,
    get_risk_levels,
    get_useful_life_params,
    get_reference_date,
)


class TestLoadConfig:
    def test_load_config_returns_dict(self):
        assert isinstance(load_config(), dict)

    def test_config_has_required_sections(self):
        config = load_config()
        required = ["app", "model_params", "device_types", "locations",
                    "hardware_states", "risk_levels", "brands"]
        for section in required:
            assert section in config, f"Missing section: {section}"


class TestModelParams:
    def test_model_params_keys(self):
        params = get_model_params()
        for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "cv_folds"]:
            assert key in params

    def test_model_params_positive(self):
        params = get_model_params()
        assert params["n_estimators"] > 0
        assert params["cv_folds"] >= 2


class TestListConfigs:
    def test_locations_are_new_campuses(self):
        assert set(get_locations()) == {"Park", "Granados", "Colon"}

    def test_hardware_states_has_five_ordered(self):
        states = get_hardware_states()
        assert states == ["Excelente", "Bueno", "Desgastado", "Malo", "Crítico"]

    def test_risk_levels_has_five_ordered(self):
        levels = get_risk_levels()
        assert levels == ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]

    def test_device_types_non_empty(self):
        assert len(get_device_types()) == 4

    def test_brands_non_empty(self):
        assert "HP" in get_brands()

    def test_useful_life_params(self):
        params = get_useful_life_params()
        assert isinstance(params, dict)
        assert len(params) > 0


class TestReferenceDate:
    def test_reference_date_is_timestamp(self):
        assert isinstance(get_reference_date(), pd.Timestamp)
