import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from librosa import load

from genre_model import __version__ as _version
from genre_model.config.core import config
from genre_model.processing.data_manager import load_pipeline
from genre_model.processing.feature_extraction import extract_features
from genre_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
pipeline = load_pipeline(file_name=pipeline_file_name)


def resolve_prediction(input_data: np.int64) -> str:
    aval_classes = config.ml_model_config.classes

    return aval_classes[input_data]


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> t.Tuple[dict, t.Optional[dict]]:

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)

    results: t.Dict[str, t.Any] = {"preds": None, "probs": None, "version": _version}

    if not errors:

        preds = pipeline.predict(X=validated_data[config.ml_model_config.features])
        probs = pipeline.predict_proba(X=validated_data[config.ml_model_config.features])

        results["preds"] = [pred for pred in preds]
        results["probs"] = [prob for prob in probs]

    return results, errors


if __name__ == "__main__":

    audio, sr = load("../GTA_ Vice City - Wildstyle  Hashim - Al Naafiysh (The Soul).mp3")
    features_data = extract_features(audio, sr)
    features_data["chroma_stft_mean"] = [np.nan]
    print(make_prediction(input_data=features_data))
