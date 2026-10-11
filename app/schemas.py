from typing import Optional
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    checking_status: str = Field(default="0 - 200 DM")
    duration: int = Field(default=18, ge=1, le=360)
    credit_history: str = Field(default="existing credits paid duly till now")
    purpose: str = Field(default="radio/television")
    credit_amount: float = Field(default=2500.0, ge=1.0)
    savings_status: str = Field(default="< 100 DM")
    employment: str = Field(default="1 - 4 years")
    installment_rate: int = Field(default=2, ge=1, le=10)
    personal_status: str = Field(default="male : single")
    other_parties: str = Field(default="none")
    residence_since: int = Field(default=2, ge=1, le=10)
    property_magnitude: str = Field(default="car or other")
    age: int = Field(default=35, ge=16, le=120)
    other_payment_plans: str = Field(default="none")
    housing: str = Field(default="own")
    existing_credits: int = Field(default=1, ge=1, le=20)
    job: str = Field(default="skilled employee / official")
    num_dependents: int = Field(default=1, ge=0, le=20)
    own_telephone: str = Field(default="yes")
    foreign_worker: str = Field(default="yes")
    model_name: Optional[str] = Field(default="random_forest")


class ModelSelectRequest(BaseModel):
    model_name: str
