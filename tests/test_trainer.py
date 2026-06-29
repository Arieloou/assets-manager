"""Tests for features/model/trainer.py — ModelTrainer class."""

import numpy as np
import pytest

from features.data import DataLoader, Preprocessor
from features.model.trainer import ModelTrainer


def _build_xy(df):
    pre = Preprocessor()
    X = pre.build_features(DataLoader.get_features(df), fit=True)
    y = pre.encode_target(DataLoader.get_targets(df), fit=True)
    return X, y, pre


class TestModelTrainerInit:
    def test_creates_model(self):
        trainer = ModelTrainer()
        assert trainer.model is not None
        assert trainer._feature_importances is None


class TestTrain:
    def test_train_sets_feature_importances(self, trained_model_and_preprocessor):
        trainer, preprocessor = trained_model_and_preprocessor
        importances = trainer._feature_importances
        assert importances is not None
        assert len(importances) == len(preprocessor.get_feature_names())

    def test_train_returns_self(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        assert trainer.train(X, y) is trainer

    def test_predict_returns_encoded_labels(self, trained_model_and_preprocessor, sample_equipos_df):
        trainer, preprocessor = trained_model_and_preprocessor
        X = preprocessor.build_features(DataLoader.get_features(sample_equipos_df), fit=False)
        preds = trainer.predict(X)
        assert len(preds) == len(sample_equipos_df)
        assert set(np.unique(preds)).issubset(set(range(5)))


class TestCrossValidate:
    def test_cv_returns_dict_with_keys(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        cv = trainer.cross_validate(X, y, k=5)
        assert set(cv) == {"mean_accuracy", "std_accuracy", "fold_scores"}
        assert len(cv["fold_scores"]) == 5

    def test_cv_accuracy_in_range(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        cv = trainer.cross_validate(X, y, k=5)
        assert 0.0 <= cv["mean_accuracy"] <= 1.0


class TestGetFeatureImportance:
    def test_importance_keys_match_feature_names(self, trained_model_and_preprocessor):
        trainer, preprocessor = trained_model_and_preprocessor
        importance = trainer.get_feature_importance()
        assert isinstance(importance, dict)
        assert set(importance.keys()) == set(preprocessor.get_feature_names())

    def test_importance_values_sum_to_one(self, trained_model_and_preprocessor):
        trainer, _ = trained_model_and_preprocessor
        total = sum(trainer.get_feature_importance().values())
        assert abs(total - 1.0) < 0.01


class TestTuneHyperparameters:
    def test_returns_self_and_sets_best_params(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        result = trainer.tune_hyperparameters(X, y, n_iter=3, cv=3)
        assert result is trainer
        assert isinstance(trainer.best_params_, dict)
        assert trainer.best_cv_score_ is not None

    def test_model_is_fitted_and_predicts(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        trainer.tune_hyperparameters(X, y, n_iter=3, cv=3)
        preds = trainer.predict(X)
        assert len(preds) == len(X)

    def test_feature_importance_available_after_tuning(self, sample_equipos_df):
        X, y, preprocessor = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        trainer.tune_hyperparameters(X, y, n_iter=3, cv=3)
        importance = trainer.get_feature_importance()
        assert set(importance.keys()) == set(preprocessor.get_feature_names())

    def test_balanced_class_weight_is_preserved(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        trainer.tune_hyperparameters(X, y, n_iter=3, cv=3)
        assert trainer.model.class_weight == 'balanced'


class TestTrainWithoutUsefulLife:
    def test_robustness_variant_trains(self, sample_equipos_df):
        X, y, _ = _build_xy(sample_equipos_df)
        trainer = ModelTrainer()
        trainer.train_without_useful_life(X, y)
        importance = trainer.get_feature_importance()
        assert "useful_life_consumed_days" not in importance


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, trained_model_and_preprocessor, tmp_path):
        trainer, _ = trained_model_and_preprocessor
        model_path = tmp_path / "test_model.joblib"
        trainer.save_model(str(model_path))

        new_trainer = ModelTrainer()
        new_trainer.load_model(str(model_path))

        original = trainer.get_feature_importance()
        loaded = new_trainer.get_feature_importance()
        for key in original:
            assert abs(original[key] - loaded[key]) < 1e-6
