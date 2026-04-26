#!/usr/bin/env python3
"""
Evaluation script (multi-model, config-driven)
"""

import argparse
import json
import time
import logging
from typing import List
import yaml

from src.models import Case, Study

logging.basicConfig(level=logging.WARNING)


# -----------------------------
# Model registry
# -----------------------------
MODEL_RUNNERS = {}


def register_model(name):
    def wrapper(fn):
        MODEL_RUNNERS[name] = fn
        return fn
    return wrapper


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
# XGBoost runner
# -----------------------------
@register_model("xgboost")
def run_xgboost(data: dict, config: dict):

    from src.xgboost_pipeline import run_full_pipeline

    truth_dict = build_truth(data)

    cases = [
        Case(
            case_id=c["case_id"],
            patient_id=c["patient_id"],
            patient_name=c.get("patient_name"),
            current_study=Study(**c["current_study"]),
            prior_studies=[Study(**s) for s in c["prior_studies"]],
        )
        for c in data["cases"]
    ]

    output_bundle = run_full_pipeline(
        cases,
        truth_dict,
    )
    output = output_bundle["model_output"]

    return output


# -----------------------------
# Placeholder: Logistic Regression
# -----------------------------
@register_model("logreg")
def run_logreg(data: dict, config: dict):

    from src.logregression_pipeline import run_logregression_pipeline

    truth_dict = build_truth(data)

    cases = [
        Case(
            case_id=c["case_id"],
            patient_id=c["patient_id"],
            patient_name=c.get("patient_name"),
            current_study=Study(**c["current_study"]),
            prior_studies=[Study(**s) for s in c["prior_studies"]],
        )
        for c in data["cases"]
    ]

    return run_logregression_pipeline(cases, truth_dict, config=config)


# -----------------------------
# Placeholder: Rules-based model
# -----------------------------
@register_model("rules")
def run_rules(data: dict, config: dict):

    from src.classifier import RelevanceClassifier, run_rules_based_pipeline
    classifier = RelevanceClassifier(threshold=config["threshold"])

    truth_dict = build_truth(data)

    cases = [
        Case(
            case_id=c["case_id"],
            patient_id=c["patient_id"],
            patient_name=c.get("patient_name"),
            current_study=Study(**c["current_study"]),
            prior_studies=[Study(**s) for s in c["prior_studies"]],
        )
        for c in data["cases"]
    ]

    return run_rules_based_pipeline(cases, truth_dict, config, classifier)


# -----------------------------
# Dispatcher
# -----------------------------
def run_model(data: dict, config: dict):

    model_type = config["experiment"]["model_type"]

    if model_type not in MODEL_RUNNERS:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            f"Available: {list(MODEL_RUNNERS.keys())}"
        )

    return MODEL_RUNNERS[model_type](data, config)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(data: dict, predictions: List[dict]):

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

    correct = incorrect = skipped = 0
    tp = tn = fp = fn = 0

    for key, gt in labeled.items():

        if key not in pred_map:
            skipped += 1
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    base_rate = sum(labeled.values()) / len(labeled) if labeled else 0

    return {
        "labeled": len(labeled),
        "predicted": len(pred_map),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "base_rate": base_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "skipped": skipped,
    }


# -----------------------------
# Pretty print
# -----------------------------
def print_results(title: str, results: dict):

    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1       : {results['f1']:.4f}")
    print(f"Base rate: {results['base_rate']:.4f}")

    print("-" * 60)
    print(
        f"TP={results['tp']} TN={results['tn']} "
        f"FP={results['fp']} FN={results['fn']}"
    )
    print(f"Skipped: {results['skipped']}")
    print("=" * 60)


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Model type: {config['experiment']['model_type']}")
    print(f"Loading data: {args.data}")

    data = load_eval_data(args.data)

    print(
        f"Cases: {len(data['cases'])} | "
        f"Truth rows: {len(data['truth'])}"
    )

    t0 = time.time()

    print("Running model...")
    output = run_model(data, config)

    print(f"Done in {time.time() - t0:.2f}s")

    # ONLY evaluate TEST split
    results = evaluate(data, output["test_predictions"])

    print_results("TEST SET EVALUATION", results)


if __name__ == "__main__":
    main()