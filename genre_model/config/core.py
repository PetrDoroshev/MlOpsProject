from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import genre_model

# Project Directories
PACKAGE_ROOT = Path(genre_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
TESTS_DIR = PACKAGE_ROOT.parent / "tests"
LOG_DIR = PACKAGE_ROOT / "logs"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    le_save_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    classes: List[str]
    features: List[str]
    variables_to_drop: Sequence[str]
    categorical_vars: Sequence[str]
    numerical_vars: Sequence[str]
    test_size: float
    random_state: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    ml_model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    import importlib.resources as pkg_resources

    tr = pkg_resources.files("genre_model").joinpath("config.yml")

    with pkg_resources.as_file(tr) as real_path:
        if real_path.is_file():
            return real_path
        raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

    # if CONFIG_FILE_PATH.is_file():
    #     return CONFIG_FILE_PATH
    # raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: Optional[YAML] = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        ml_model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
