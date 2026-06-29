import pandas as pd

from features.config import get_risk_levels


class ModelPredictor:
    """Predicts ``operational_risk_level`` and its confidence (soft output)."""

    def __init__(self, model, preprocessor):
        """Initialize with a trained model and a fitted preprocessor.

        Args:
            model: Object with ``predict``/``predict_proba`` (ModelTrainer or
                a raw sklearn pipeline).
            preprocessor: Fitted ``Preprocessor`` instance.
        """
        self.model = model
        self.preprocessor = preprocessor

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the encoded feature matrix for the given raw rows."""
        return self.preprocessor.build_features(df, fit=False)

    def predict(self, input_dict):
        """Predict the operational risk level for a single device record.

        Args:
            input_dict: Raw feature values (acquisition_date,
                technical_incident_rate, last_reactive_maintenance_date,
                last_preventive_maintenance_date, hardware_integrity_status,
                headquarters_location, device_type, device_brand).

        Returns:
            The predicted ``operational_risk_level`` label (string).
        """
        X = self._prepare(pd.DataFrame([input_dict]))
        encoded = self.model.predict(X)[0]
        return self.preprocessor.decode_target([encoded])[0]

    def predict_proba(self, input_dict):
        """Return the soft output: vote proportion per risk level.

        Args:
            input_dict: Raw feature values for a single record.

        Returns:
            Dict ``{nivel_riesgo: proporcion}`` ordered low -> high risk, where
            the proportions sum to ~1.0 (interpreted as prediction confidence).
        """
        X = self._prepare(pd.DataFrame([input_dict]))
        proba = self.model.predict_proba(X)[0]

        # Map model class indices (encoded ordinals) back to risk labels.
        classes = self.model.classes_
        labels = self.preprocessor.decode_target(classes)
        proba_by_label = dict(zip(labels, proba))

        # Return ordered by the configured risk order, filling missing classes.
        return {level: float(proba_by_label.get(level, 0.0)) for level in get_risk_levels()}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict risk levels for a DataFrame of raw records.

        Returns the input DataFrame with an ``operational_risk_level`` column.
        """
        df = df.copy()
        X = self._prepare(df)
        encoded = self.model.predict(X)
        df['operational_risk_level'] = self.preprocessor.decode_target(encoded)
        return df
