from src.classifier import score_prior
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def build_train_and_test_set(cases, truth_dict):
    X, y, ids, groups = [], [], [], []


    for case in cases:
        current = case.current_study
        case_id = case.case_id

        for prior in case.prior_studies:
            score, debug = score_prior(current, prior)

            features = [
                int(debug["final_score"] >= 0.5),
                debug["final_score"],
                debug["anatomy_score"],
                debug["modality_score"],
                debug["recency_score"],
                int(debug["cur_anatomy"] == debug["pri_anatomy"]),
                int(debug["cur_modality"] == debug["pri_modality"]),
                int(debug["modality_score"] == 0),
            ]

            key = (case_id, prior.study_id)
            label = truth_dict.get(key)

            if label is None:
                continue

            X.append(features)
            y.append(label)
            ids.append(key)
            groups.append(case_id)   # 👈 THIS IS THE KEY FIX

    print("Total candidate pairs:", sum(len(c.prior_studies) for c in cases))
    print("Labeled pairs:", len(X))

    return X, y, ids, groups



def find_threshold_for_target_recall(y_true, y_prob, target_recall=0.75):
    thresholds = np.arange(0.0, 1.01, 0.01)

    best_threshold = 0.5
    best_f1 = -1
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)

        # only consider thresholds that meet recall requirement
        if recall >= target_recall:
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_metrics = {
                    "recall": recall,
                    "precision": precision,
                    "f1": f1
                }

    return best_threshold, best_metrics

def find_best_threshold(y_true, y_prob, optimize_for="f1"):
    """
    Find the best classification threshold.

    Args:
        y_true: ground truth labels (0/1)
        y_prob: predicted probabilities
        optimize_for: "f1" or "recall"

    Returns:
        best_threshold, metrics_at_best_threshold
    """

    thresholds = np.arange(0.1, 0.9, 0.05)

    best_threshold = 0.5
    best_score = -1
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        if optimize_for == "recall":
            score = recall
        else:
            score = f1  # default

        if score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = {
                "f1": f1,
                "recall": recall,
                "precision": precision
            }

    return best_threshold, best_metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_logregression(X, y, ids, groups):
    # --------------------------
    # Split for proper evaluation
    # --------------------------
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]

    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]

    ids_train = [ids[i] for i in train_idx]
    ids_test  = [ids[i] for i in test_idx]

    # --------------------------
    # Train model
    # --------------------------
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # --------------------------
    # Get test probabilities
    # --------------------------
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------
    # Tune threshold
    # --------------------------
    best_threshold, best_metrics = find_threshold_for_target_recall(
        y_test,
        y_test_prob,
    )

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Metrics at best threshold: {best_metrics}")

    # --------------------------
    # Apply tuned threshold (TEST)
    # --------------------------
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    test_predictions = [
        {
            "case_id": cid,
            "study_id": sid,
            "predicted_is_relevant": bool(pred),
            "probability": float(prob),
        }
        for (cid, sid), pred, prob in zip(ids_test, y_test_pred, y_test_prob)
    ]

    # --------------------------
    # Apply tuned threshold (FULL DATA)
    # --------------------------
    y_full_prob = model.predict_proba(X)[:, 1]
    y_full_pred = (y_full_prob >= best_threshold).astype(int)

    full_predictions = [
        {
            "case_id": cid,
            "study_id": sid,
            "predicted_is_relevant": bool(pred),
            "probability": float(prob),
        }
        for (cid, sid), pred, prob in zip(ids, y_full_pred, y_full_prob)
    ]

    # --------------------------
    # return everything you need
    # --------------------------
    return {
        "model": model,
        "threshold": best_threshold,
        "metrics": best_metrics,
        "test_predictions": test_predictions,
        "full_predictions": full_predictions,
        "test_truth": list(zip(ids_test, y_test)),
    }