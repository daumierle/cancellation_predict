from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Drop is_canceled column for predict"""
    if "is_canceled" in input_data.columns:
        validated_data = input_data.drop(columns=["is_canceled"])
    else:
        validated_data = input_data
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleHotelDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class HotelDataInputSchema(BaseModel):
    # hotel: Optional[str]
    lead_time: Optional[int]
    length_of_stay: Optional[int]
    occupants: Optional[int]
    distribution_channel: Optional[str]
    is_repeated_guest: Optional[int]
    previous_cancellations: Optional[int]
    previous_bookings_not_canceled: Optional[int]
    booking_changes: Optional[int]
    deposit_type: Optional[str]
    days_in_waiting_list: Optional[int]
    customer_type: Optional[str]
    adr: Optional[float]
    required_car_parking_spaces: Optional[int]
    total_of_special_requests: Optional[int]


class MultipleHotelDataInputs(BaseModel):
    inputs: List[HotelDataInputSchema]

