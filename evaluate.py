#!/usr/bin/env python3
"""
Local evaluation script.

Usage:
    python evaluate.py --data public_eval.json [--url http://localhost:8000/predict]

If --url is not provided, runs the classifier directly in-process (no server needed).
"""

import argparse
import json
import sys
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.WARNING)

def load_eval_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_local(data: dict) -> List[dict]:
    """Run classifier in-process without a server."""
    # Add src to path
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from src.classifier import RelevanceClassifier
    from src.models import Case, Study, ChallengeRequest

    classifier = RelevanceClassifier()
    predictions = []

    for case_data in data["cases"]:
        case = Case(
            case_id=case_data["case_id"],
            patient_id=case_data["patient_id"],
            patient_name=case_data.get("patient_name"),
            current_study=Study(**case_data["current_study"]),
            prior_studies=[Study(**s) for s in case_data["prior_studies"]],
        )
        preds = classifier.classify_case(case)
        for p in preds:
            predictions.append({
                "case_id": p.case_id,
                "study_id": p.study_id,
                "predicted_is_relevant": p.predicted_is_relevant,
            })

    return predictions


def run_remote(data: dict, url: str) -> List[dict]:
    """Send request to the running server."""
    import httpx
    resp = httpx.post(url, json=data, timeout=360)
    resp.raise_for_status()
    return resp.json()["predictions"]


def evaluate(data: dict, predictions: List[dict]) -> dict:
    """Compute accuracy against labeled data."""
    truth = {
    (item["case_id"], item["study_id"]): item["is_relevant_to_current"]
    for item in data["truth"]
    }
    total_priors = 0

    for case in data["cases"]:
        for prior in case["prior_studies"]:
            key = (case["case_id"], prior["study_id"])
            total_priors += 1

    labeled = {k: v for k, v in truth.items() if v is not None}

    if not labeled:
        print("No labels found in eval data. Cannot compute accuracy.")
        return {}

    pred_map = {}
    for p in predictions:
        key = (p["case_id"], p["study_id"])
        pred_map[key] = p["predicted_is_relevant"]

    correct = 0
    incorrect = 0
    skipped = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for key, gt in labeled.items():
        if key not in pred_map:
            skipped += 1
            incorrect += 1
            continue

        pred = pred_map[key]
        if pred == gt:
            correct += 1
            if gt:
                true_pos += 1
            else:
                true_neg += 1
        else:
            incorrect += 1
            if pred:
                false_pos += 1
            else:
                false_neg += 1

    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    base_rate = sum(1 for v in labeled.values() if v) / len(labeled)

    return {
        "total_priors": total_priors,
        "labeled": len(labeled),
        "predicted": len(pred_map),
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "base_rate_relevant": base_rate,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
    }



def main():
    parser = argparse.ArgumentParser(description="Evaluate radiology prior relevance model")
    parser.add_argument("--data", required=True, help="Path to eval JSON file")
    parser.add_argument("--url", help="API endpoint URL (if omitted, runs in-process)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"Loading eval data from: {args.data}")
    data = load_eval_data(args.data)

    n_cases = len(data.get("cases", []))
    n_priors = sum(len(c["prior_studies"]) for c in data.get("cases", []))
    print(f"Cases: {n_cases}  |  Prior exams: {n_priors}")

    t0 = time.time()
    if args.url:
        print(f"Running against: {args.url}")
        predictions = run_remote(data, args.url)
    else:
        print("Running in-process (no server required)")
        predictions = run_local(data)
    elapsed = time.time() - t0

    print(f"Predictions received: {len(predictions)}  |  Elapsed: {elapsed:.2f}s")

    results = evaluate(data, predictions)
    if not results:
        return

    print("\n" + "=" * 50)
    print(f"  ACCURACY:   {results['accuracy']:.4f}  ({results['correct']}/{results['correct']+results['incorrect']})")
    print(f"  PRECISION:  {results['precision']:.4f}")
    print(f"  RECALL:     {results['recall']:.4f}")
    print(f"  F1:         {results['f1']:.4f}")
    print(f"  BASE RATE:  {results['base_rate_relevant']:.4f}  (fraction actually relevant)")
    print("=" * 50)
    print(f"  TP={results['true_pos']}  TN={results['true_neg']}  FP={results['false_pos']}  FN={results['false_neg']}")
    print(f"  Skipped (counted incorrect): {results['skipped']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
