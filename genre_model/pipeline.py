# from feature_engine.encoding import OrdinalEncoder
# from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from genre_model.config.core import config
from genre_model.model_init import genre_classifier

genre_pipe = Pipeline(
    [
        # ("categorical_encoder", OrdinalEncoder(variables=config.model_config.categorical_vars)),
        ("scaler", StandardScaler()),
        ("Model", genre_classifier),
    ]
)
