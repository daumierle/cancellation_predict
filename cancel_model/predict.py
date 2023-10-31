from typing import Union
import pandas as pd

from cancel_model import __version__ as _version
from cancel_model.config.core import config
from cancel_model.utils.data_utils import load_pipeline
from cancel_model.utils.features import feature_selection
from cancel_model.utils.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction_lr(input_data: Union[pd.DataFrame, dict]):
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    features = feature_selection()
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(
            X=validated_data[features]
        )
        results = {
            "predictions": predictions,
            "version": _version,
            "errors": errors,
        }

    return results
