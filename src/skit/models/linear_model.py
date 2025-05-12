import logging
from typing import Literal, Any

import numpy as np
from pydantic_numpy import np_array_pydantic_annotated_typing as pyd_type

from skit.models import Model

logger = logging.getLogger(__name__)


class LogisticRegression(Model):
    """
    Logistic Regression (logit, MaxEnt) classifier.
    """

    name: Literal["LogisticRegression"]
    w: pyd_type(np.float32) = 0.0  # type: ignore
    b: pyd_type(np.float32) = 0.0  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        for i in range(10):
            logger.info(f"Iteration {i + 1}...")
        logger.info(f"Fitting {self.name} model...")

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs: Any) -> np.ndarray:
        return (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()

    def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        logger.info(f"Predicting with {self.name} model...")
        return np.random.randint(2, size=x.shape[0])
