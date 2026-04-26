# src/features.py

import numpy as np
from src.logregression import score_prior
from src.models import Case

def build_case_features(case: Case, runtime):

    current = case.current_study
    priors = case.prior_studies

    if len(priors) == 0:
        return np.array([]), []

    embs = runtime.case_encoder.encode_case(
        current.study_description,
        [p.study_description for p in priors]
    )

    cur_raw = embs[0]
    prior_raw = embs[1:]

    if len(prior_raw) == 0:
        return np.array([]), []

    cur_norm = np.linalg.norm(cur_raw)
    prior_norms = [np.linalg.norm(p) for p in prior_raw]

    cur_emb = cur_raw / (cur_norm + 1e-8)
    prior_embs = [p / (np.linalg.norm(p) + 1e-8) for p in prior_raw]

    sims = [float(np.dot(cur_emb, p)) for p in prior_embs]

    max_sim = max(sims)
    mean_sim = float(np.mean(sims))

    X = []
    pairs = []

    for prior, pri_emb, sim, p_norm in zip(priors, prior_embs, sims, prior_norms):

        emb_diff = float(np.linalg.norm(cur_emb - pri_emb))
        sim_gap = max_sim - sim

        _, debug = score_prior(current, prior)

        days_apart = debug["days_apart"]

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

        X.append(features)
        pairs.append((case.case_id, prior.study_id))

    return np.array(X), pairs