import typing as t
import pandas as pd
from genre_model import __version__ as _version
from genre_model.processing.data_manager import load_pipeline
from genre_model.processing.validation import validate_inputs
from genre_model.processing.feature_extraction import extract_features
from librosa import load
from config.core import config

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
pipeline = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:

    data = pd.DataFrame(input_data)
    #validated_data, errors = validate_inputs(input_data=data)

    results: t.Dict[str, t.Any] = {"preds": None, "probs": None, "version": _version}

    preds = pipeline.predict(X=data[config.ml_model_config.features])
    probs = pipeline.predict_proba(X=data[config.ml_model_config.features])

    results["preds"] = [pred for pred in preds]
    results["probs"] = [prob for prob in probs]

    return results


if __name__ == "__main__":

    audio, sr = load("../GTA_ Vice City - Wildstyle  Hashim - Al Naafiysh (The Soul).mp3")
    features_data = extract_features(audio, sr)
    print(make_prediction(input_data=features_data))
