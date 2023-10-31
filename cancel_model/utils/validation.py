from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
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
    hotel: Optional[str]
    lead_time: Optional[int]
    arrival_date_year: Optional[int]
    arrival_date_month: Optional[str]
    arrival_date_week_number: Optional[int]
    arrival_date_day_of_month: Optional[int]
    stays_in_weekend_nights: Optional[int]
    stays_in_week_nights: Optional[int]
    adults: Optional[int]
    children: Optional[int]
    babies: Optional[int]
    meal: Optional[str]
    country: Optional[str]
    market_segment: Optional[str]
    distribution_channel: Optional[str]
    is_repeated_guest: Optional[int]
    previous_cancellations: Optional[int]
    previous_bookings_not_canceled: Optional[int]
    reserved_room_type: Optional[str]
    assigned_room_type: Optional[str]
    booking_changes: Optional[int]
    deposit_type: Optional[str]
    agent: Optional[int]
    company: Optional[int]
    days_in_waiting_list: Optional[int]
    customer_type: Optional[str]
    adr: Optional[float]
    required_car_parking_spaces: Optional[int]
    total_of_special_requests: Optional[int]
    reservation_status: Optional[str]
    reservation_status_date: Optional[str]


class MultipleHotelDataInputs(BaseModel):
    inputs: List[HotelDataInputSchema]

