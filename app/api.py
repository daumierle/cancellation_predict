import json
import pandas as pd
import numpy as np
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from loguru import logger
from cancel_model import __version__ as model_version
from cancel_model.predict import make_prediction_lr

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.model_dump()


@api_router.post("/predict_lr", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleHotelDataInputs) -> Any:
    """
    Make house price predictions with the logistic regression model
    """
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction_lr(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
