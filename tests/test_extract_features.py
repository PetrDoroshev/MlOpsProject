from genre_model.config.core import config
from genre_model.processing import feature_extraction


def test_extract_features_returns_dict(audio_data):

    result = feature_extraction.extract_features(*audio_data)
    assert isinstance(result, dict)


def test_extract_features_correct_structure(audio_data):

    result = feature_extraction.extract_features(*audio_data)
    assert set(result.keys()) == set(config.ml_model_config.features)
    for val in result.values():
        assert isinstance(val, list)
        assert len(val) == 1
