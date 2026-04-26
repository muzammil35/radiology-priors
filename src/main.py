"""
Radiology Prior Relevance API
Predicts whether prior examinations should be shown to a radiologist
reading a current examination.
"""

import logging
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from .classifier import RelevanceClassifier
from .models import ChallengeRequest, ChallengeResponse, Prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Radiology Prior Relevance API",
    description="Predicts whether prior radiology exams are relevant to a current exam",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = RelevanceClassifier()


@app.get("/")
def root():
    return {
        "service": "Radiology Prior Relevance API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoint": "POST /predict"
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=ChallengeResponse)
async def predict(request: ChallengeRequest, raw_request: Request):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    case_count = len(request.cases)
    prior_count = sum(len(c.prior_studies) for c in request.cases)

    logger.info(
        f"[{request_id}] Received request | challenge={request.challenge_id} "
        f"| cases={case_count} | priors={prior_count}"
    )

    predictions: List[Prediction] = []

    for case in request.cases:
        case_predictions = classifier.classify_case(case)
        predictions.extend(case_predictions)
        logger.debug(
            f"[{request_id}] case={case.case_id} "
            f"priors={len(case.prior_studies)} "
            f"relevant={sum(1 for p in case_predictions if p.predicted_is_relevant)}"
        )

    elapsed = time.time() - start
    logger.info(
        f"[{request_id}] Done | predictions={len(predictions)} | elapsed={elapsed:.3f}s"
    )

    return ChallengeResponse(predictions=predictions)
