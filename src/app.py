# app.py

from fastapi import FastAPI
import joblib
import numpy as np

from src.models import Case, RequestPayload
from src.features import build_case_features
from src.inference_runtime import runtime

app = FastAPI()

bundle = joblib.load("model_bundle.pkl")
model = bundle["model"]
threshold = bundle["threshold"]

from typing import List

@app.post("/predict")
def predict(payload: RequestPayload):

    predictions = []

    for case in payload.cases:

        X, pairs = build_case_features(case, runtime)

        if len(X) == 0:
            continue  # don't return early, just skip

        probs = model.predict_proba(X)[:, 1]
        preds = probs >= threshold

        for (cid, sid), p, pred in zip(pairs, probs, preds):
            predictions.append({
                "case_id": cid,
                "study_id": sid,
                "predicted_is_relevant": bool(pred),
            })

    return {
        "predictions": predictions
    }