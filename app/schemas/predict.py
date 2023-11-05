from typing import Optional, Any, List
from pydantic import BaseModel
from cancel_model.utils.validation import HotelDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultipleHotelDataInputs(BaseModel):
    inputs: List[HotelDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [{
                    "lead_time": 100,
                    "length_of_stay": 3,
                    "occupants": 2,
                    "distribution_channel": "Direct",
                    "is_repeated_guest": 0,
                    "previous_cancellations": 0,
                    "previous_bookings_not_canceled": 2,
                    "booking_changes": 1,
                    "deposit_type": "Non Refund",
                    "days_in_waiting_list": 0,
                    "customer_type": "Transient",
                    "adr": 105.2,
                    "required_car_parking_spaces": 1,
                    "total_of_special_requests": 1
                }]
            }
        }
