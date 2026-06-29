import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features.config import (
    get_hardware_states,
    get_risk_levels,
    get_reference_date,
    get_locations,
    get_device_types,
    get_brands,
)


class Preprocessor:
    """Feature engineering, encoding and scaling for operational-risk prediction.

    Pipeline:
      1. ``engineer_features`` derives day-difference features from the raw
         acquisition / maintenance dates.
      2. The four quantitative features (the three day-based features plus
         ``technical_incident_rate``) are scaled with a ``StandardScaler``.
      3. ``hardware_integrity_status`` is encoded with an ``OrdinalEncoder``
         (best -> worst).
      4. ``headquarters_location``, ``device_type`` and ``device_brand`` are
         encoded with a ``OneHotEncoder``.
      5. The target ``operational_risk_level`` is encoded with an
         ``OrdinalEncoder`` (low -> high risk).
    """

    # Quantitative features scaled with StandardScaler.
    QUANTITATIVE_FEATURES = [
        'useful_life_consumed_days',
        'technical_incident_rate',
        'days_since_last_corrective_maintenance',
        'days_since_last_preventive_maintenance',
    ]
    ORDINAL_FEATURE = 'hardware_integrity_status'
    ONEHOT_FEATURES = ['headquarters_location', 'device_type', 'device_brand']
    TARGET = 'operational_risk_level'

    def __init__(self, reference_date=None):
        self.reference_date = reference_date
        self.hardware_order = get_hardware_states()
        self.risk_order = get_risk_levels()

        self.scaler = None           # StandardScaler for the quantitative features (``sc``)
        self.status_encoder = None   # OrdinalEncoder for hardware_integrity_status
        self.onehot_encoder = None   # OneHotEncoder for location + type + brand
        self.target_encoder = None   # OrdinalEncoder for operational_risk_level
        self._feature_names = None

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    def _ref_date(self):
        return self.reference_date if self.reference_date is not None else get_reference_date()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive day-difference features from the date columns.

        Adds ``useful_life_consumed_days``, ``days_since_last_corrective_maintenance``
        and ``days_since_last_preventive_maintenance`` (days elapsed up to the
        reference date). When ``last_reactive_maintenance_date`` is missing (no
        corrective maintenance ever performed), the days-since-corrective feature
        falls back to ``useful_life_consumed_days``.
        """
        df = df.copy()
        ref = pd.Timestamp(self._ref_date())

        acquisition = pd.to_datetime(df['acquisition_date'], dayfirst=True, errors='coerce')
        corrective = pd.to_datetime(df['last_reactive_maintenance_date'], dayfirst=True, errors='coerce')
        preventive = pd.to_datetime(df['last_preventive_maintenance_date'], dayfirst=True, errors='coerce')

        df['useful_life_consumed_days'] = (ref - acquisition).dt.days
        df['days_since_last_corrective_maintenance'] = (ref - corrective).dt.days
        df['days_since_last_preventive_maintenance'] = (ref - preventive).dt.days

        # No corrective maintenance -> it has gone its whole life without one.
        df['days_since_last_corrective_maintenance'] = (
            df['days_since_last_corrective_maintenance'].fillna(df['useful_life_consumed_days'])
        )

        return df

    # ------------------------------------------------------------------ #
    # Feature matrix
    # ------------------------------------------------------------------ #
    def build_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Build the encoded, scaled feature matrix ready for the model.

        Args:
            df: DataFrame with the raw feature columns (dates, status, location,
                type, brand, technical_incident_rate).
            fit: If True, fit the scaler/encoders; otherwise only transform.

        Returns:
            DataFrame with scaled quantitative, numeric, ordinal and one-hot columns.
        """
        df = self.engineer_features(df)

        # Quantitative features -> StandardScaler (``sc``)
        quant = df[self.QUANTITATIVE_FEATURES].astype(float)
        if fit:
            self.scaler = StandardScaler()
            quant_scaled = self.scaler.fit_transform(quant)
        else:
            quant_scaled = self.scaler.transform(quant)
        quant_df = pd.DataFrame(quant_scaled, columns=self.QUANTITATIVE_FEATURES)

        # Ordinal feature: hardware_integrity_status
        if fit:
            self.status_encoder = OrdinalEncoder(categories=[self.hardware_order])
            status_arr = self.status_encoder.fit_transform(df[[self.ORDINAL_FEATURE]])
        else:
            status_arr = self.status_encoder.transform(df[[self.ORDINAL_FEATURE]])
        status_df = pd.DataFrame(status_arr, columns=['hardware_integrity_status_ord'])

        # One-hot features: headquarters_location, device_type, device_brand.
        # Categories are pinned from config so the feature space is stable
        # regardless of which categories appear in a given training batch.
        if fit:
            self.onehot_encoder = OneHotEncoder(
                categories=[get_locations(), get_device_types(), get_brands()],
                handle_unknown='ignore',
                sparse_output=False,
            )
            onehot_arr = self.onehot_encoder.fit_transform(df[self.ONEHOT_FEATURES])
        else:
            onehot_arr = self.onehot_encoder.transform(df[self.ONEHOT_FEATURES])
        onehot_cols = list(self.onehot_encoder.get_feature_names_out(self.ONEHOT_FEATURES))
        onehot_df = pd.DataFrame(onehot_arr, columns=onehot_cols)

        X = pd.concat([quant_df, status_df, onehot_df], axis=1)

        if fit:
            self._feature_names = list(X.columns)
        return X

    # ------------------------------------------------------------------ #
    # Target
    # ------------------------------------------------------------------ #
    def encode_target(self, target, fit: bool = True) -> np.ndarray:
        """Encode ``operational_risk_level`` with an OrdinalEncoder.

        Args:
            target: DataFrame, Series or array-like of target labels.
            fit: If True, fit the encoder; otherwise only transform.

        Returns:
            1D numpy array of integer-encoded risk levels.
        """
        if isinstance(target, pd.DataFrame):
            values = target[[self.TARGET]] if self.TARGET in target.columns else target.iloc[:, [0]]
        elif isinstance(target, pd.Series):
            values = target.to_frame(name=self.TARGET)
        else:
            values = pd.DataFrame({self.TARGET: np.asarray(target).ravel()})

        if fit:
            self.target_encoder = OrdinalEncoder(categories=[self.risk_order])
            encoded = self.target_encoder.fit_transform(values)
        else:
            encoded = self.target_encoder.transform(values)

        return encoded.ravel().astype(int)

    def decode_target(self, encoded) -> np.ndarray:
        """Inverse-transform encoded risk levels back to their labels."""
        arr = np.asarray(encoded).reshape(-1, 1).astype(float)
        return self.target_encoder.inverse_transform(arr).ravel()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_feature_names(self):
        """Return the expanded feature-matrix column names (after fit)."""
        return list(self._feature_names) if self._feature_names is not None else []
