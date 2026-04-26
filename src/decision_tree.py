
from xgboost import XGBClassifier
from src.logistic_regression import find_threshold_for_target_recall, score_prior
import numpy as np
from typing import List
from src.models import Case
from src.clinical_bert import ClinicalBERTEncoder, CaseEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import optuna

def build_train_and_test_set(cases: List[Case], truth_dict):
    X, y, ids, groups = [], [], [], []

    encoder = ClinicalBERTEncoder("emilyalsentzer/Bio_ClinicalBERT")
    case_encoder = CaseEncoder(encoder)

    for case in cases:
        current = case.current_study
        case_id = case.case_id
        priors = case.prior_studies

        if len(priors) == 0:
            continue

        # ---------------------------------------------------
        # ONE BATCH ENCODING PER CASE (CRITICAL FOR SPEED)
        # ---------------------------------------------------
        texts = [current.study_description] + [
            p.study_description for p in priors
        ]

        embs = case_encoder.encode_case(
            current.study_description,
            [p.study_description for p in priors]
        )

        cur_emb = embs[0]
        prior_embs = embs[1:]

        # ---------------------------------------------------
        # PRECOMPUTE CONTEXT FEATURES (IMPORTANT FIX)
        # ---------------------------------------------------
        sims = [float(np.dot(cur_emb, p)) for p in prior_embs]

        max_sim = max(sims)
        mean_sim = float(np.mean(sims))

        for i, (prior, pri_emb, sim) in enumerate(zip(priors, prior_embs, sims)):

            # -----------------------------
            # CORE SEMANTIC FEATURES
            # -----------------------------
            emb_diff = float(np.linalg.norm(cur_emb - pri_emb))
            sim_gap = max_sim - sim

            # -----------------------------
            # OPTIONAL HEURISTIC FEATURES (SAFE NOW)
            # -----------------------------
            score, debug = score_prior(current, prior)

            features = [

                # PRIMARY SIGNAL (ClinicalBERT)
                sim,
                max_sim,
                sim_gap,
                mean_sim,
                emb_diff,

                # temporal signal
                debug["days_apart"],
                debug["recency_score"],

                # weak structured signals (kept but de-emphasized)
                debug["anatomy_score"],
                debug["modality_score"],

                # stability features
                int(debug["days_apart"] < 30),
                int(debug["days_apart"] < 365),
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
# OPTUNA OBJECTIVE
# =========================================================
def _objective(trial, X, y, groups):

    params = {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
    }

    gkf = GroupKFold(n_splits=3)
    scores = []

    for train_idx, val_idx in gkf.split(X, y, groups):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        probs = model.predict_proba(X_val)[:, 1]

        thresh, metrics = find_threshold_for_target_recall(
            y_val, probs, target_recall=0.65
        )

        scores.append(metrics.get("precision", 0))

    return float(np.mean(scores))

# =========================================================
# OPTIONAL: HYPERPARAM TUNING
# =========================================================
def tune_xgboost(X, y, groups, n_trials=50):

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: _objective(trial, X, y, groups),
        n_trials=n_trials
    )

    print("Best params:", study.best_params)

    return study

def train_xgboost(X, y, ids, groups):

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ids_train = [ids[i] for i in train_idx]
    ids_test = [ids[i] for i in test_idx]

    if best_params is None:
        best_params = dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=5,
            reg_lambda=5.0,
            reg_alpha=1.0,
            eval_metric="logloss",
            random_state=42,
        )

    # ---------------------------------------------------
    # STRONGER REGULARIZATION (FIXES PRECISION COLLAPSE)
    # ---------------------------------------------------
    model = XGBClassifier(**best_params)

    model.fit(X_train, y_train,eval_set=[(X_test, y_test)],
    verbose=False)

    # --------------------------
    # probabilities
    # --------------------------
    y_test_prob = model.predict_proba(X_test)[:, 1]

    best_threshold, best_metrics = find_threshold_for_best_accuracy(y_test, y_test_prob)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Metrics: {best_metrics}")

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
    # full dataset predictions
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

    return {
        "model": model,
        "threshold": best_threshold,
        "metrics": best_metrics,
        "test_predictions": test_predictions,
        "full_predictions": full_predictions,
    }


from sklearn.metrics import precision_score, recall_score, f1_score

def cross_validate_xgboost(X, y, groups, params, n_splits=5):

    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        probs = model.predict_proba(X_val)[:, 1]

        threshold, metrics = find_threshold_for_target_recall(
            y_val, probs, target_recall=0.65
        )

        preds = (probs >= threshold).astype(int)

        fold_result = {
            "fold": fold,
            "precision": precision_score(y_val, preds),
            "recall": recall_score(y_val, preds),
            "f1": f1_score(y_val, preds),
            "threshold": threshold,
        }

        fold_results.append(fold_result)

        print(f"Fold {fold}: {fold_result}")

    # ------------------------
    # summary
    # ------------------------
    summary = {
        "precision_mean": np.mean([f["precision"] for f in fold_results]),
        "recall_mean": np.mean([f["recall"] for f in fold_results]),
        "f1_mean": np.mean([f["f1"] for f in fold_results]),
        "precision_std": np.std([f["precision"] for f in fold_results]),
        "recall_std": np.std([f["recall"] for f in fold_results]),
        "f1_std": np.std([f["f1"] for f in fold_results]),
    }

    print("\nCV SUMMARY:", summary)

    return summary, fold_results


def find_threshold_for_best_accuracy(y_true, y_prob):
    thresholds = np.linspace(0, 1, 200)
    best_thresh, best_acc = 0.5, 0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, {"accuracy": best_acc}

