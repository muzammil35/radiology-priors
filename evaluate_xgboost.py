#!/usr/bin/env python3
"""
Local evaluation script for XGBoost model.

Usage:
    python evaluate_xgboost.py --data public_eval.json
"""

import argparse
import json
import sys
import time
import logging
from typing import List

from src.models import Case, Study

logging.basicConfig(level=logging.WARNING)


# -----------------------------
# Load data
# -----------------------------
def load_eval_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# -----------------------------
# Build truth dict
# -----------------------------
def build_truth(data: dict):
    return {
        (item["case_id"], item["study_id"]): item["is_relevant_to_current"]
        for item in data["truth"]
    }


# -----------------------------
# Run XGBoost model locally
# -----------------------------
def run_xgboost(data: dict):
    """
    Runs your XGBoost model in-process.
    """

    import os
    sys.path.insert(0, os.path.dirname(__file__))


    from src.decision_trees import  train_xgboost, build_train_and_test_set

    truth_dict = build_truth(data)

    cases = []

    for case_data in data["cases"]:
        cases.append(Case(
            case_id=case_data["case_id"],
            patient_id=case_data["patient_id"],
            patient_name=case_data.get("patient_name"),
            current_study=Study(**case_data["current_study"]),
            prior_studies=[Study(**s) for s in case_data["prior_studies"]],
        ))

    X, y, ids, groups = build_train_and_test_set(cases, truth_dict)

    output = train_xgboost(X, y, ids, groups)

    return output


# -----------------------------
# Evaluation (unchanged logic)
# -----------------------------
def evaluate(data: dict, predictions: List[dict]) -> dict:
    truth = {
        (item["case_id"], item["study_id"]): item["is_relevant_to_current"]
        for item in data["truth"]
    }

    
    pred_map = {
        (p["case_id"], p["study_id"]): p["predicted_is_relevant"]
        for p in predictions
    }

    labeled = {
        k: truth[k]
        for k in pred_map.keys()
        if k in truth and truth[k] is not None
    }

    total_priors = sum(len(case["prior_studies"]) for case in data["cases"])

    correct = incorrect = skipped = 0
    tp = tn = fp = fn = 0

    for key, gt in labeled.items():
        if key not in pred_map:
            skipped += 1
            incorrect += 1
            continue

        pred = pred_map[key]

        if pred == gt:
            correct += 1
            if gt:
                tp += 1
            else:
                tn += 1
        else:
            incorrect += 1
            if pred:
                fp += 1
            else:
                fn += 1

    accuracy = correct / (correct + incorrect) if (correct + incorrect) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    base_rate = sum(labeled.values()) / len(labeled)

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
        "true_pos": tp,
        "true_neg": tn,
        "false_pos": fp,
        "false_neg": fn,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument("--data", required=True, help="Path to eval JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"Loading eval data from: {args.data}")
    data = load_eval_data(args.data)

    n_cases = len(data.get("cases", []))
    n_priors = sum(len(c["prior_studies"]) for c in data.get("cases", []))
    print(f"Cases: {n_cases} | Prior exams: {n_priors}")

    t0 = time.time()

    print("Running XGBoost locally...")
    output = run_xgboost(data)

    elapsed = time.time() - t0
    print(f"Predictions: {len(output['full_predictions'])} | Time: {elapsed:.2f}s")

    print("Evaluating FULL dataset...")
    results = evaluate(data, output["full_predictions"])

    if not results:
        return

    print("\n" + "=" * 50)
    print(f"  ACCURACY:   {results['accuracy']:.4f}")
    print(f"  PRECISION:  {results['precision']:.4f}")
    print(f"  RECALL:     {results['recall']:.4f}")
    print(f"  F1:         {results['f1']:.4f}")
    print(f"  BASE RATE:  {results['base_rate_relevant']:.4f}")
    print("=" * 50)
    print(f"  TP={results['true_pos']} TN={results['true_neg']} "
          f"FP={results['false_pos']} FN={results['false_neg']}")
    print(f"  Skipped: {results['skipped']}")
    print("=" * 50)

    print("Evaluating TEST set...")
    results = evaluate(data, output["test_predictions"])

    print("\n" + "=" * 50)
    print(f"  ACCURACY:   {results['accuracy']:.4f}")
    print(f"  PRECISION:  {results['precision']:.4f}")
    print(f"  RECALL:     {results['recall']:.4f}")
    print(f"  F1:         {results['f1']:.4f}")
    print(f"  BASE RATE:  {results['base_rate_relevant']:.4f}")
    print("=" * 50)
    print(f"  TP={results['true_pos']} TN={results['true_neg']} "
          f"FP={results['false_pos']} FN={results['false_neg']}")
    print(f"  Skipped: {results['skipped']}")
    print("=" * 50)


if __name__ == "__main__":
    main()