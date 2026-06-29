import joblib
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from features.config import get_model_params


class ModelTrainer:
    """Random Forest trainer for the single target ``operational_risk_level``.

    The quantitative features are already standardized by the ``Preprocessor``
    (its ``scaler`` / ``sc``), so the model itself is a bare
    ``RandomForestClassifier`` (``random_classifier_model``).
    """

    def __init__(self):
        params = get_model_params()
        self.model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            criterion='entropy',
            class_weight='balanced',
            random_state=50,
        )
        self._feature_importances = None
        self._feature_names = None
        self.best_params_ = None
        self.best_cv_score_ = None

    # ------------------------------------------------------------------ #
    def _capture(self, X):
        """Store the feature names from the training matrix when available."""
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)

    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train):
        """Fit the model on the training data and return self."""
        self._capture(X_train)
        self.model.fit(X_train, y_train)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict(self, X):
        """Predict encoded risk levels for the given feature matrix."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return per-class vote proportions (soft output)."""
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_

    def cross_validate(self, X, y, k=5):
        """Perform k-fold cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=k)
        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }

    def tune_hyperparameters(self, X, y, n_iter=None, cv=None, scoring='f1_macro', random_state=50):
        """Search for better hyperparameters with ``RandomizedSearchCV``.

        Searches ``n_estimators``, ``max_depth``, ``min_samples_split``,
        ``min_samples_leaf``, ``max_features`` and ``criterion``, keeping
        ``class_weight='balanced'`` fixed to address the risk-level class
        imbalance. ``scoring='f1_macro'`` is used instead of accuracy so the
        search isn't biased toward the majority risk classes.

        Sets ``self.model`` to the best estimator found and records
        ``self.best_params_`` / ``self.best_cv_score_``. Returns self.
        """
        params = get_model_params()
        n_iter = n_iter if n_iter is not None else params.get('n_iter_search', 25)
        cv = cv if cv is not None else params.get('cv_folds', 5)

        param_distributions = {
            'n_estimators': randint(100, 400),
            'max_depth': [None, 5, 8, 10, 12, 15, 20],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 6),
            'max_features': ['sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],
        }

        base_model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X, y)

        self._capture(X)
        self.model = search.best_estimator_
        self._feature_importances = self.model.feature_importances_
        self.best_params_ = search.best_params_
        self.best_cv_score_ = float(search.best_score_)
        return self

    def get_feature_importance(self):
        """Return feature importances as a {name: importance} dict."""
        if self._feature_importances is None:
            self._feature_importances = self.model.feature_importances_

        names = self._feature_names
        if names is None or len(names) != len(self._feature_importances):
            names = [f"feature_{i}" for i in range(len(self._feature_importances))]
        return dict(zip(names, self._feature_importances))

    def train_without_useful_life(self, X, y):
        """Train a variant excluding ``useful_life_consumed_days`` (robustness check).

        Per ``Proyecto IA.md``, this checks the model is not over-reliant on a
        single potentially-leaky feature.
        """
        params = get_model_params()
        model_wo = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            criterion='entropy',
            class_weight='balanced',
            random_state=50,
        )

        if hasattr(X, 'columns') and 'useful_life_consumed_days' in X.columns:
            X_wo = X.drop(columns=['useful_life_consumed_days'])
        else:
            X_wo = np.delete(np.asarray(X), 0, axis=1)

        self._capture(X_wo)
        model_wo.fit(X_wo, y)
        self.model = model_wo
        self._feature_importances = model_wo.feature_importances_
        return self

    def save_model(self, filepath):
        """Persist the model (and feature names) with joblib."""
        joblib.dump({"model": self.model, "feature_names": self._feature_names}, filepath)

    def load_model(self, filepath):
        """Load a model previously saved with ``save_model``."""
        data = joblib.load(filepath)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self._feature_names = data.get("feature_names")
        else:
            self.model = data
        self._feature_importances = self.model.feature_importances_
