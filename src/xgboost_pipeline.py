from xgboost import XGBClassifier
from src.classifier import score_prior
import numpy as np
from typing import List, Optional, Dict, Any
from src.models import Case
from src.clinical_bert import ClinicalBERTEncoder, CaseEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import optuna
import joblib


# =========================================================
# SAFE CONFIG HELPER
# =========================================================
def _cfg(config, *path, default=None):
    """Safely access nested dict config."""
    cur = config or {}
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# =========================================================
# FEATURE BUILDING (UNCHANGED)
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

        cur_raw = embs[0]
        prior_raw = embs[1:]

        if len(prior_raw) == 0:
            continue

        cur_norm = np.linalg.norm(cur_raw)
        prior_norms = [np.linalg.norm(p) for p in prior_raw]

        cur_emb = cur_raw / (cur_norm + 1e-8)
        prior_embs = [p / (np.linalg.norm(p) + 1e-8) for p in prior_raw]

        sims = [float(np.dot(cur_emb, p)) for p in prior_embs]

        if len(sims) == 0:
            continue

        max_sim = max(sims)
        mean_sim = float(np.mean(sims))

        for prior, pri_emb, sim, p_norm in zip(priors, prior_embs, sims, prior_norms):

            emb_diff = float(np.linalg.norm(cur_emb - pri_emb))
            sim_gap = max_sim - sim

            _, debug = score_prior(current, prior)

            features = [
                sim,
                p_norm,
                cur_norm,
                abs(cur_norm - p_norm),
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
# OPTUNA OBJECTIVE (CONFIG DRIVEN)
# =========================================================
def objective(trial, X, y, groups, config=None):

    model_cfg = _cfg(config, "model", default={})

    params = {
        "n_estimators": model_cfg.get("n_estimators", 500),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            model_cfg.get("learning_rate", {}).get("min", 0.01),
            model_cfg.get("learning_rate", {}).get("max", 0.2),
            log=model_cfg.get("learning_rate", {}).get("log", True),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            model_cfg.get("max_depth", {}).get("min", 2),
            model_cfg.get("max_depth", {}).get("max", 6),
        ),
        "subsample": trial.suggest_float(
            "subsample",
            model_cfg.get("subsample", {}).get("min", 0.5),
            model_cfg.get("subsample", {}).get("max", 1.0),
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            model_cfg.get("colsample_bytree", {}).get("min", 0.5),
            model_cfg.get("colsample_bytree", {}).get("max", 1.0),
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            model_cfg.get("min_child_weight", {}).get("min", 1),
            model_cfg.get("min_child_weight", {}).get("max", 10),
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            model_cfg.get("reg_lambda", {}).get("min", 0.01),
            model_cfg.get("reg_lambda", {}).get("max", 10.0),
            log=model_cfg.get("reg_lambda", {}).get("log", True),
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            model_cfg.get("reg_alpha", {}).get("min", 0.001),
            model_cfg.get("reg_alpha", {}).get("max", 5.0),
            log=model_cfg.get("reg_alpha", {}).get("log", True),
        ),
        "eval_metric": model_cfg.get("eval_metric", "logloss"),
        "random_state": model_cfg.get("random_state", 42),
    }

    n_splits = _cfg(config, "cv", "n_splits", default=3)
    gkf = GroupKFold(n_splits=n_splits)

    scores = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx], verbose=False)

        probs = model.predict_proba(X[val_idx])[:, 1]

        from sklearn.metrics import roc_auc_score
        scores.append(roc_auc_score(y[val_idx], probs))

    return float(np.mean(scores))


# =========================================================
# TUNING
# =========================================================
def tune_xgboost(X, y, groups, config=None):

    n_trials = _cfg(config, "optuna", "n_trials", default=50)
    direction = _cfg(config, "optuna", "direction", default="maximize")

    study = optuna.create_study(direction=direction)

    study.optimize(
        lambda t: objective(t, X, y, groups, config),
        n_trials=n_trials
    )

    print("\nBEST PARAMS:", study.best_params)
    return study


# =========================================================
# THRESHOLD SELECTION
# =========================================================
def find_best_threshold(y_true, probs, config=None):

    strategy = _cfg(config, "threshold", "strategy", default="accuracy")

    min_t = _cfg(config, "threshold", "search_range", "min", default=0.1)
    max_t = _cfg(config, "threshold", "search_range", "max", default=0.9)
    steps = _cfg(config, "threshold", "search_range", "steps", default=100)

    thresholds = np.linspace(min_t, max_t, steps)

    best_t, best_score = 0.5, -1

    for t in thresholds:
        preds = (probs >= t).astype(int)

        if strategy == "f1":
            score = f1_score(y_true, preds)
        else:
            score = accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


# =========================================================
# CROSS VALIDATION
# =========================================================
def cross_validate_xgboost(X, y, groups, params, config=None):

    n_splits = _cfg(config, "cv", "n_splits", default=5)
    gkf = GroupKFold(n_splits=n_splits)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

        model = XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx], verbose=False)

        train_probs = model.predict_proba(X[train_idx])[:, 1]
        val_probs = model.predict_proba(X[val_idx])[:, 1]

        threshold, _ = find_best_threshold(y[train_idx], train_probs, config)

        val_preds = (val_probs >= threshold).astype(int)
        f1 = f1_score(y[val_idx], val_preds)

        scores.append(f1)
        print(f"Fold {fold}: F1={f1:.4f}, threshold={threshold:.3f}")

    return {
        "f1_mean": np.mean(scores),
        "f1_std": np.std(scores),
    }, scores


# =========================================================
# FINAL TRAINING
# =========================================================
def train_xgboost(X, y, ids, groups, best_params, config=None):

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    train_probs = model.predict_proba(X_train)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_train, train_probs, config)

    save_model(model, best_threshold)

    print(f"Best threshold (train): {best_threshold:.3f}, F1={best_f1:.4f}")

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    acc = accuracy_score(y_test, test_preds)

    print("\nFINAL TEST ACCURACY:", acc)

    test_predictions = [
        {
            "case_id": cid,
            "study_id": sid,
            "predicted_is_relevant": bool(pred),
            "probability": float(prob),
        }
        for (cid, sid), pred, prob in zip(
            [ids[i] for i in test_idx],
            test_preds,
            test_probs
        )
    ]

    return {
        "model": model,
        "threshold": best_threshold,
        "test_accuracy": acc,
        "test_predictions": test_predictions,
    }


# =========================================================
# FULL PIPELINE
# =========================================================
def run_full_pipeline(cases, truth_dict, config: Optional[Dict[str, Any]] = None):

    X, y, ids, groups = build_train_and_test_set(cases, truth_dict)

    study = tune_xgboost(X, y, groups, config=config)

    best_params = dict(study.best_params)

    print("\nRunning cross-validation with best params...")
    cv_summary, _ = cross_validate_xgboost(
        X, y, groups,
        params=best_params,
        config=config
    )

    print("\nTraining final model...")
    output = train_xgboost(
        X, y, ids, groups,
        best_params,
        config=config
    )

    return {
        "model_output": output,
        "cv_summary": cv_summary,
        "best_params": best_params,
        "threshold": output["threshold"],
    }


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model, threshold, path="model_bundle.pkl"):
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
        },
        path
    )