from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pydantic import BaseModel


class Model(BaseModel, ABC):
    """
    Abstract class for all models.
    """

    name: str
    random_state: int

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass
