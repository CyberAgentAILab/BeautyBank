from typing import Annotated

import numpy as np
from pydantic import PlainValidator
from pydantic_core.core_schema import ValidationInfo


def validate_ndarray(v: np.ndarray, info: ValidationInfo) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        raise ValueError(f"Expected np.ndarray, got {type(v)}")
    return v


NdArray = Annotated[np.ndarray, PlainValidator(validate_ndarray)]
