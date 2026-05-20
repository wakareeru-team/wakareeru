from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionWithThreshold(BaseEstimator, ClassifierMixin):
    """Logistic regression wrapper that stores the chosen prediction threshold."""

    def __init__(self, estimator, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        prob = self.estimator.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def register_legacy_main_alias() -> None:
    """Allow loading models pickled when stage_12 was run as __main__."""
    import __main__

    __main__.LogisticRegressionWithThreshold = LogisticRegressionWithThreshold
