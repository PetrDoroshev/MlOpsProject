import pytest
from genre_model.config.core import config, TESTS_DIR
from genre_model.processing.data_manager import load_dataset
from librosa import load
from pathlib import Path
import os

@pytest.fixture()
def input_data():
    return load_dataset(file_name=config.app_config.test_data_file)

@pytest.fixture
def audio_data():
    test_file = TESTS_DIR / "classical.00005.wav"
    return load(test_file)
