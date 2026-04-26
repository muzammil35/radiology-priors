from xgboost import XGBClassifier
from src.logistic_regression import score_prior
import numpy as np
from typing import List
from src.models import Case
from src.clinical_bert import ClinicalBERTEncoder, CaseEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import optuna
from sklearn.metrics import f1_score


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

        # --- raw embeddings
        cur_raw = embs[0]
        prior_raw = embs[1:]

        if len(prior_raw) == 0:
            continue

        # --- norms (NEW SIGNAL)
        cur_norm = np.linalg.norm(cur_raw)
        prior_norms = [np.linalg.norm(p) for p in prior_raw]

        # --- normalize (cosine similarity)
        cur_emb = cur_raw / (cur_norm + 1e-8)
        prior_embs = [p / (np.linalg.norm(p) + 1e-8) for p in prior_raw]

        # --- cosine similarities
        sims = [float(np.dot(cur_emb, p)) for p in prior_embs]

        if len(sims) == 0:
            continue

        max_sim = max(sims)
        mean_sim = float(np.mean(sims))

        for prior, pri_emb, sim, p_norm in zip(priors, prior_embs, sims, prior_norms):

            emb_diff = float(np.linalg.norm(cur_emb - pri_emb))
            sim_gap = max_sim - sim

            # --- existing metadata features
            _, debug = score_prior(current, prior)

            features = [
                sim,                          # cosine similarity
                p_norm,                       # NEW: prior embedding magnitude
                cur_norm,                     # NEW: current embedding magnitude
                abs(cur_norm - p_norm),       # NEW: magnitude mismatch
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
# OPTUNA OBJECTIVE (NOW MATCHES DEPLOYMENT EXACTLY)
# =========================================================
def objective(trial, X, y, groups):

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

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, probs)
        scores.append(auc)

    return float(np.mean(scores))


# =========================================================
# OPTUNA TUNING
# =========================================================
def tune_xgboost(X, y, groups, n_trials=50):

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X, y, groups), n_trials=n_trials)

    print("\nBEST PARAMS:", study.best_params)
    return study


# =========================================================
# CROSS VALIDATION (ALSO SWITCHED TO REAL METRIC)
# =========================================================
def cross_validate_xgboost(X, y, groups, params, n_splits=5):

    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        # --- get probs
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]

        # --- learn threshold ONLY from train split
        threshold, train_f1 = find_best_threshold(y_train, train_probs)

        # --- apply to validation
        val_preds = (val_probs >= threshold).astype(int)

        f1 = f1_score(y_val, val_preds)
        scores.append(f1)

        print(f"Fold {fold}: F1={f1:.4f}, threshold={threshold:.3f}")

    summary = {
        "f1_mean": np.mean(scores),
        "f1_std": np.std(scores),
    }

    print("\nCV SUMMARY:", summary)
    return summary, scores

from sklearn.metrics import accuracy_score

def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.1, 0.9, 100)
    best_t, best_acc = 0.5, 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)

        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t, best_acc


# =========================================================
# FINAL TRAINING (SIMPLIFIED + CLEAN)
# =========================================================
def train_xgboost(X, y, ids, groups, best_params ):

    X = np.array(X)
    y = np.array(y)

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    # get validation-style threshold from TRAIN split (or use CV if you want cleaner)
    train_probs = model.predict_proba(X_train)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_train, train_probs)

    save_model(model, best_threshold)

    print(f"Best threshold (train): {best_threshold:.3f}, F1={best_f1:.4f}")

    # apply to test
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_test_pred)

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
            y_test_pred,
            y_test_prob
        )
    ]

    return {
        "model": model,
        "threshold": best_threshold,
        "test_accuracy": acc,
        "test_predictions": test_predictions,
    }


# =========================================================
# PIPELINE
# =========================================================
def run_full_pipeline(cases, truth_dict):

    X, y, ids, groups = build_train_and_test_set(cases, truth_dict)

    study = tune_xgboost(X, y, groups, n_trials=30)

    best_params = {
        k: v for k, v in study.best_params.items()
        if k != "threshold"
    }
   
    print("\nRunning cross-validation with best params...")
    cv_summary, _ = cross_validate_xgboost(
        X, y, groups,
        params=best_params,
        n_splits=5
    )

    print("\nTraining final model...")
    output = train_xgboost(
        X, y, ids, groups, best_params,
    )

    return {
        "model_output": output,
        "cv_summary": cv_summary,
        "best_params": best_params,
        "threshold": output["threshold"],
    }


import joblib

def save_model(model, threshold, path="model_bundle.pkl"):
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
        },
        path
    )