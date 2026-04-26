from xgboost import XGBClassifier
from src.logregression import find_threshold_for_target_recall, score_prior
import numpy as np
from typing import List, Tuple, Dict, Any
from src.models import Case
from src.clinical_bert import ClinicalBERTEncoder, CaseEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import optuna


# =========================================================
# FEATURE BUILDING
# =========================================================
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

        embs = case_encoder.encode_case(
            current.study_description,
            [p.study_description for p in priors]
        )

        cur_emb = embs[0]
        prior_embs = embs[1:]

        sims = [float(np.dot(cur_emb, p)) for p in prior_embs]

        if len(sims) == 0:
            continue

        max_sim = max(sims)
        mean_sim = float(np.mean(sims))

        for prior, pri_emb, sim in zip(priors, prior_embs, sims):

            emb_diff = float(np.linalg.norm(cur_emb - pri_emb))
            sim_gap = max_sim - sim

            _, debug = score_prior(current, prior)

            features = [
                sim,
                max_sim,
                sim_gap,
                mean_sim,
                emb_diff,

                debug["days_apart"],
                debug["recency_score"],
                debug["anatomy_score"],
                debug["modality_score"],

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

    return np.array(X), np.array(y), ids, np.array(groups)


# =========================================================
# OPTUNA OBJECTIVE (ACCURACY-BASED)
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

        best_thresh, _ = find_threshold_for_best_accuracy(y_val, probs)

        preds = (probs >= best_thresh).astype(int)

        acc = accuracy_score(y_val, preds)
        scores.append(acc)

    return float(np.mean(scores))


# =========================================================
# OPTUNA TUNING
# =========================================================
def tune_xgboost(X, y, groups, n_trials=50):

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: _objective(trial, X, y, groups),
        n_trials=n_trials
    )

    print("\nBEST PARAMS:", study.best_params)

    return study


# =========================================================
# CROSS VALIDATION REPORT
# =========================================================
def cross_validate_xgboost(X, y, groups, params, n_splits=5):

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        probs = model.predict_proba(X_val)[:, 1]

        best_thresh, _ = find_threshold_for_best_accuracy(y_val, probs)

        preds = (probs >= best_thresh).astype(int)

        acc = accuracy_score(y_val, preds)

        fold_results.append({
            "fold": fold,
            "accuracy": acc,
            "threshold": best_thresh
        })

        print(f"Fold {fold}: accuracy={acc:.4f}, threshold={best_thresh:.2f}")

    summary = {
        "accuracy_mean": np.mean([f["accuracy"] for f in fold_results]),
        "accuracy_std": np.std([f["accuracy"] for f in fold_results]),
    }

    print("\nCV SUMMARY:", summary)

    return summary, fold_results


# =========================================================
# FINAL TRAINING
# =========================================================
def train_xgboost(X, y, ids, groups, best_params=None):

    X = np.array(X)
    y = np.array(y)

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

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

    model = XGBClassifier(**best_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # --------------------------
    # threshold tuning
    # --------------------------
    y_test_prob = model.predict_proba(X_test)[:, 1]

    best_threshold, best_metrics = find_threshold_for_best_accuracy(
        y_test,
        y_test_prob
    )

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Metrics: {best_metrics}")

    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    test_predictions = [
        {
            "case_id": cid,
            "study_id": sid,
            "predicted_is_relevant": bool(pred),
            "probability": float(prob),
        }
        for (cid, sid), pred, prob in zip(
            [ids[i] for i in test_idx],
            y_test_pred,
            y_test_prob
        )
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


# =========================================================
# THRESHOLD SEARCH
# =========================================================
def find_threshold_for_best_accuracy(y_true, y_prob):
    thresholds = np.linspace(0, 1, 200)

    best_thresh = 0.5
    best_acc = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        acc = (preds == y_true).mean()

        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_thresh, {"accuracy": best_acc}


def run_full_pipeline(cases, truth_dict):

    # 1. build dataset
    X, y, ids, groups = build_train_and_test_set(cases, truth_dict)

    # 2. hyperparameter tuning (THIS WAS MISSING)
    study = tune_xgboost(X, y, groups, n_trials=30)

    best_params = study.best_params

    print("\nBest hyperparameters from Optuna:")
    print(best_params)

    # 3. cross-validation report (THIS WAS ALSO MISSING)
    print("\nRunning cross-validation with best params...")
    cv_summary, fold_results = cross_validate_xgboost(
        X, y, groups,
        params=best_params,
        n_splits=5
    )

    # 4. final training
    print("\nTraining final model...")
    output = train_xgboost(X, y, ids, groups, best_params=best_params)

    return {
        "model_output": output,
        "cv_summary": cv_summary,
        "fold_results": fold_results,
        "best_params": best_params
    }