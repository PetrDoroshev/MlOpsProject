import logging
import pickle
from pathlib import Path

from config.core import LOG_DIR, config
from pipeline import genre_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from genre_model import __version__ as _version
from genre_model.config.core import TRAINED_MODEL_DIR


def run_training() -> None:
    """Train the model."""
    # Update logs
    log_path = Path(f"{LOG_DIR}/log_{_version}.log")
    if Path.exists(log_path):
        log_path.unlink()
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    le = LabelEncoder()

    data.drop(labels=config.ml_model_config.variables_to_drop, axis=1, inplace=True)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.ml_model_config.features],  # predictors
        data[config.ml_model_config.target],
        test_size=config.ml_model_config.test_size,
        random_state=config.ml_model_config.random_state,
    )

    y_lb_train = le.fit_transform(y_train)

    path_to_save = f"{TRAINED_MODEL_DIR}/{config.app_config.le_save_file}"
    with open(path_to_save, "wb") as f:
        pickle.dump(le, f)

    y_lb_test = le.transform(y_test)

    # fit model
    genre_pipe.fit(X_train, y_lb_train)

    # make predictions for train set
    class_ = genre_pipe.predict(X_train)
    # pred = genre_pipe.predict_proba(X_train)[:, 1]

    # determine train accuracy and roc-auc
    train_accuracy = accuracy_score(y_lb_train, class_)
    # train_roc_auc = roc_auc_score(y_train, pred)

    print(f"train accuracy: {train_accuracy}")
    # print(f"train roc-auc: {train_roc_auc}")
    print()

    logging.info(f"train accuracy: {train_accuracy}")
    # logging.info(f"train roc-auc: {train_roc_auc}")

    # make predictions for test set
    class_ = genre_pipe.predict(X_test)
    # pred = genre_pipe.predict_proba(X_test)[:, 1]

    # determine test accuracy and roc-auc
    test_accuracy = accuracy_score(y_lb_test, class_)
    # test_roc_auc = roc_auc_score(y_test, pred)

    print(f"test accuracy: {test_accuracy}")
    # print(f"test roc-auc: {test_roc_auc}")
    print()

    logging.info(f"test accuracy: {test_accuracy}")
    # logging.info(f"test roc-auc: {test_roc_auc}")

    # persist trained model
    save_pipeline(pipeline_to_persist=genre_pipe)


if __name__ == "__main__":
    run_training()
