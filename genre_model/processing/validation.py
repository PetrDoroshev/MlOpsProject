import re
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from genre_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    assert input_data.columns.tolist() == config.ml_model_config.features

    relevant_data = input_data[config.ml_model_config.features].copy()

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleAudioInputs(inputs=relevant_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return relevant_data, errors


class AudioInputSchema(BaseModel):
    chroma_stft_mean: float
    chroma_stft_var: float
    rms_mean: float
    rms_var: float
    spectral_centroid_mean: float
    spectral_centroid_var: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_var: float
    rolloff_mean: float
    rolloff_var: float
    zero_crossing_rate_mean: float
    zero_crossing_rate_var: float
    harmony_mean: float
    harmony_var: float
    tempo: float
    mfcc1_mean: float
    mfcc1_var: float
    mfcc2_mean: float
    mfcc2_var: float
    mfcc3_mean: float
    mfcc3_var: float
    mfcc4_mean: float
    mfcc4_var: float
    mfcc5_mean: float
    mfcc5_var: float
    mfcc6_mean: float
    mfcc6_var: float
    mfcc7_mean: float
    mfcc7_var: float
    mfcc8_mean: float
    mfcc8_var: float
    mfcc9_mean: float
    mfcc9_var: float
    mfcc10_mean: float
    mfcc10_var: float
    mfcc11_mean: float
    mfcc11_var: float
    mfcc12_mean: float
    mfcc12_var: float
    mfcc13_mean: float
    mfcc13_var: float
    mfcc14_mean: float
    mfcc14_var: float
    mfcc15_mean: float
    mfcc15_var: float
    mfcc16_mean: float
    mfcc16_var: float
    mfcc17_mean: float
    mfcc17_var: float
    mfcc18_mean: float
    mfcc18_var: float
    mfcc19_mean: float
    mfcc19_var: float
    mfcc20_mean: float
    mfcc20_var: float

class MultipleAudioInputs(BaseModel):
    inputs: List[AudioInputSchema]
