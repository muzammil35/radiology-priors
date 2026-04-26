from src.classifier import score_prior
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score


# =========================================================
# FEATURE BUILDING
# =========================================================

def build_train_and_test_set(cases, truth_dict, config):
    X, y, ids, groups = [], [], [], []

    score_threshold = config["features"]["score_threshold"]

    for case in cases:
        current = case.current_study
        case_id = case.case_id

        for prior in case.prior_studies:
            _, debug = score_prior(current, prior)

            features = [
                int(debug["final_score"] >= score_threshold),
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
            groups.append(case_id)

    print("Total candidate pairs:", sum(len(c.prior_studies) for c in cases))
    print("Labeled pairs:", len(X))

    return X, y, ids, groups


# =========================================================
# THRESHOLD SEARCH (CONFIG DRIVEN)
# =========================================================

def find_best_threshold(y_true, y_prob, config):
    cfg = config["threshold"]

    thresholds = np.linspace(
        cfg["search_range"]["min"],
        cfg["search_range"]["max"],
        cfg["search_range"]["steps"]
    )

    best_threshold = 0.5
    best_score = -1
    best_metrics = {}

    optimize_for = cfg["strategy"]

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)

        if optimize_for == "recall":
            score = recall
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = {
                "f1": f1,
                "recall": recall,
                "precision": precision,
            }

    return best_threshold, best_metrics


# =========================================================
# MODEL BUILDER (CONFIG DRIVEN)
# =========================================================

def build_logreg_model(config):
    model_cfg = config["model"]

    params = {
        "max_iter": model_cfg.get("max_iter", 1000),
        "class_weight": model_cfg.get("class_weight", "balanced"),
    }

    if "solver" in model_cfg:
        params["solver"] = model_cfg["solver"]

    if "C" in model_cfg:
        params["C"] = model_cfg["C"]

    if "penalty" in model_cfg:
        params["penalty"] = model_cfg["penalty"]

    return LogisticRegression(**params)


# =========================================================
# PIPELINE
# =========================================================

def run_logregression_pipeline(cases, truth_dict, config):

    # --------------------------
    # Build dataset
    # --------------------------
    X, y, ids, groups = build_train_and_test_set(cases, truth_dict, config)

    # --------------------------
    # Group split
    # --------------------------
    split_cfg = config["cv"]

    gss = GroupShuffleSplit(
        test_size=0.2,
        random_state=split_cfg.get("random_state", 42)
    )

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    def idx_subset(arr, idx):
        return [arr[i] for i in idx]

    X_train = idx_subset(X, train_idx)
    X_test = idx_subset(X, test_idx)

    y_train = idx_subset(y, train_idx)
    y_test = idx_subset(y, test_idx)

    ids_test = idx_subset(ids, test_idx)

    # --------------------------
    # Train model (CONFIG DRIVEN)
    # --------------------------
    model = build_logreg_model(config)
    model.fit(X_train, y_train)

    # --------------------------
    # Predict probabilities
    # --------------------------
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------
    # Threshold tuning
    # --------------------------
    best_threshold, best_metrics = find_best_threshold(
        y_test,
        y_test_prob,
        config
    )

    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Metrics: {best_metrics}")

    # --------------------------
    # Predictions helper
    # --------------------------
    def make_predictions(ids_list, probs, preds):
        return [
            {
                "case_id": cid,
                "study_id": sid,
                "predicted_is_relevant": bool(p),
                "probability": float(prob),
            }
            for (cid, sid), p, prob in zip(ids_list, preds, probs)
        ]

    # --------------------------
    # TEST SET
    # --------------------------
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    test_predictions = make_predictions(
        ids_test,
        y_test_prob,
        y_test_pred
    )

    # --------------------------
    # FULL DATA
    # --------------------------
    y_full_prob = model.predict_proba(X)[:, 1]
    y_full_pred = (y_full_prob >= best_threshold).astype(int)

    full_predictions = make_predictions(
        ids,
        y_full_prob,
        y_full_pred
    )

    # --------------------------
    # RETURN
    # --------------------------
    return {
        "model": model,
        "threshold": best_threshold,
        "metrics": best_metrics,
        "test_predictions": test_predictions,
        "full_predictions": full_predictions,
        "test_truth": list(zip(ids_test, y_test)),
    }