from typing import Union

import numpy as np
from pydantic import BaseModel, Field

from skit.models import LogisticRegression


class Job(BaseModel):
    model: Union[LogisticRegression] = Field(..., discriminator="name")

    def run(self) -> np.ndarray:
        x_train, x_test, y_train, y_test = ..., ..., ..., ...  # noqa: F841
        self.model.fit(x_train, y_train)
        return self.model.predict(np.zeros(2))
