import numpy as np
import pandas as pd

from genre_model.predict import make_prediction


def test_make_prediction(input_data):

    results, errors = make_prediction(input_data=input_data)
    assert isinstance(results, dict)
    assert errors is None or isinstance(errors, dict)
