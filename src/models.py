from pydantic import BaseModel
from typing import List, Optional


class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str  # ISO date string


class Case(BaseModel):
    case_id: str
    patient_id: str
    patient_name: Optional[str] = None
    current_study: Study
    prior_studies: List[Study]


class ChallengeRequest(BaseModel):
    challenge_id: str
    schema_version: int = 1
    generated_at: Optional[str] = None
    cases: List[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class ChallengeResponse(BaseModel):
    predictions: List[Prediction]
