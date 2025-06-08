import os
from pathlib import Path

import pytest
from librosa import load

from genre_model.config.core import TESTS_DIR, config
from genre_model.processing.data_manager import load_dataset


@pytest.fixture()
def input_data():
    temp_dataset = load_dataset(file_name=config.app_config.test_data_file)
    temp_dataset.drop(labels=config.ml_model_config.variables_to_drop, axis=1, inplace=True)

    return temp_dataset[config.ml_model_config.features]


@pytest.fixture
def audio_data():
    test_file = TESTS_DIR / "classical.00005.wav"
    return load(test_file)
