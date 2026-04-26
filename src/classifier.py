"""
Rule-based relevance classifier for radiology prior studies.

Decision logic:
1. Extract modality and anatomy from study descriptions
2. Score each prior based on modality match, anatomy match, and recency
3. Apply threshold to produce binary prediction

This approach runs in microseconds per study with no external dependencies,
eliminating timeout risk on large batches.
"""

import re
import logging
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict
from functools import lru_cache

from .models import Case, Prediction, Study

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modality definitions
# Each entry: canonical name -> list of regex patterns to match
# ---------------------------------------------------------------------------
MODALITY_PATTERNS: Dict[str, List[str]] = {
    "MRI": [r"\bMRI\b", r"\bMR\b", r"\bMAGNETIC\b", r"\bFMRI\b", r"\bDWI\b", r"\bDTI\b", r"\bMRCP\b", r"\bMRA\b"],
    # NM must come before CT so "PET CT" matches NM, not CT
    "NM": [r"\bPET\b", r"\bNUCLEAR\b", r"\bNM\b", r"\bSPECT\b", r"\bSCINTI\b", r"\bGALLIUM\b", r"\bBONE SCAN\b", r"\bMIBI\b"],
    "CT": [r"\bCT\b", r"\bCAT\b", r"\bCOMPUTED TOM", r"\bCTA\b"],
    "XRAY": [r"\bX[\-\s]?RAY\b", r"\bRADIOGRAPH", r"\bXR\b", r"\bPORTABLE\b", r"\bCHEST PA\b", r"\bCHEST AP\b", r"\bABD\b.*\bKUB\b", r"\bKUB\b"],
    "US": [r"\bULTRA?SOUND\b", r"\b\bUS\b", r"\bECHO\b", r"\bDOPPLER\b", r"\bSONO\b"],

    "FLUORO": [r"\bFLUORO\b", r"\bBARIUM\b", r"\bSWALLOW\b", r"\bENEMA\b", r"\bMYELO\b", r"\bARTHROGRAM\b"],
    "MAMMO": [r"\bMAMMO\b", r"\bMAMMOGRAM\b", r"\bTOMO\b.*\bBREAST\b", r"\bBREAST\b.*\bTOMO\b"],
    "DEXA": [r"\bDEXA\b", r"\bDXA\b", r"\bBONE DENSITY\b", r"\bBMD\b"],
    "IR": [r"\bBIOPSY\b", r"\bDRAINAGE\b", r"\bINTERVENT\b", r"\bABLATION\b", r"\bEMBOLI\b"],
    "ANGIO": [r"\bANGIO\b", r"\bARTERIOGRAM\b", r"\bVENOGRAM\b"],
}


VIEW_PATTERNS = [
    r"\b\d+\s*VIEW",
    r"\bFRONTAL\b",
    r"\bLATERAL\b",
    r"\bLAT\b",
    r"\bAP\b",
    r"\bPA\b"
]


def heuristic_modality(desc: str):
    d = desc.upper()

    if any(re.search(p, d) for p in VIEW_PATTERNS):
        return "XRAY"

    if "CHEST" in d and "CT" not in d:
        return "XRAY"

    if "HIP" in d or "KNEE" in d:
        if "VIEW" in d:
            return "XRAY"

    return None

# Modality compatibility matrix: if prior modality is in the set for a given
# current modality, it's considered a modality-compatible pair
MODALITY_COMPAT: Dict[str, set] = {
    "MRI":    {"MRI", "CT", "NM", "US"},
    "CT":     {"CT", "MRI", "XRAY", "NM"},
    "XRAY":   {"XRAY", "CT", "FLUORO"},
    "US":     {"US", "CT", "MRI"},
    "NM":     {"NM", "CT", "MRI"},
    "FLUORO": {"FLUORO", "XRAY", "CT"},
    "MAMMO":  {"MAMMO", "US", "MRI"},
    "DEXA":   {"DEXA", "XRAY"},
    "IR":     {"IR", "CT", "US", "ANGIO"},
    "ANGIO":  {"ANGIO", "CT", "MRI", "IR"},
}


DEFAULT_MODALITY_BY_REGION = {
    "chest": "XRAY",
    "spine": "XRAY",
    "upper_ext": "XRAY",
    "lower_ext": "XRAY",
    "abdomen": "CT"
}

# ---------------------------------------------------------------------------
# Anatomy definitions
# Grouped by clinical region — same group = anatomy match
# ---------------------------------------------------------------------------
ANATOMY_GROUPS: Dict[str, List[str]] = {
    "brain":        ["brain", "head", "cranial", "cranium", "cerebr", "intracran", "skull",
                     "orbit", "sella", "pituitary", "posterior fossa", "stroke", "tia",
                     "neuro", "cerebellum", "temporal lobe", "frontal lobe", "iac"],
    "spine_c":      ["cervical", "c-spine", "c spine", "neck"],
    "spine_t":      ["thoracic", "t-spine", "t spine"],
    "spine_l":      ["lumbar", "l-spine", "l spine", "lumbosacral", "ls spine"],
    "spine_s":      ["sacrum", "sacral", "coccyx", "tailbone"],
    "chest":        ["chest", "lung", "pulmonary", "thorax", "thoracic", "mediastin",
                     "pleural", "pericardial", "cardiac", "heart", "aorta", "pe ",
                     "pulm embol"],
    "abdomen":      ["abdomen", "abdominal", "abd", "liver", "hepatic", "biliary",
                     "gallbladder", "pancreas", "pancreatic", "spleen", "splenic",
                     "kidney", "renal", "adrenal", "bowel", "colon", "appendix",
                     "retroperitoneal", "kub"],
    "pelvis":       ["pelvis", "pelvic", "bladder", "uterus", "uterine", "ovary",
                     "ovarian", "prostate", "rectum", "rectal", "hip", "sacroiliac",
                     "si joint"],
    "breast":       ["breast", "mammo"],
    "upper_ext":    ["shoulder", "humerus", "elbow", "forearm", "wrist", "hand",
                     "finger", "upper extremity", "upper ext", "ue "],
    "lower_ext":    ["hip", "femur", "knee", "tibia", "fibula", "ankle", "foot",
                     "toe", "lower extremity", "lower ext", "le "],
    "whole_body":   ["whole body", "total body", "skeleton", "bone scan"],
    "vascular":     ["aorta", "carotid", "iliac", "femoral artery", "peripheral vasc",
                     "runoff", "mesenteric", "celiac", "angio"],
    "face_neck":    ["face", "facial", "sinus", "sinuses", "mandible", "jaw", "orbit",
                     "tmo", "tmj", "parotid", "thyroid", "soft tissue neck", "neck"],
}

# Pre-flatten anatomy terms for faster lookup
_ANATOMY_FLAT: Dict[str, str] = {}  # term -> group_name
for group, terms in ANATOMY_GROUPS.items():
    for term in terms:
        _ANATOMY_FLAT[term] = group


@lru_cache(maxsize=4096)
def extract_modality(description: str) -> Optional[str]:
    """Extract the primary imaging modality from a study description."""
    desc = description.upper()
    for modality, patterns in MODALITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, desc):
                return modality
    return None or heuristic_modality(description)


@lru_cache(maxsize=4096)
def extract_anatomy(description: str) -> Optional[str]:
    """Extract the primary anatomical region from a study description."""
    desc = description.lower()
    # Longer terms first to avoid partial matches
    for term in sorted(_ANATOMY_FLAT.keys(), key=len, reverse=True):
        if term in desc:
            return _ANATOMY_FLAT[term]
    return None



def parse_date(date_str: str) -> Optional[date]:
    """Parse ISO date string, return None on failure."""
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_between(d1: Optional[date], d2: Optional[date]) -> Optional[int]:
    if d1 is None or d2 is None:
        return None
    return abs((d1 - d2).days)


def fallback_modality(desc, cur_anatomy):
    heuristic_mod = heuristic_modality(desc)
    if not heuristic_mod:
        if cur_anatomy:
            return DEFAULT_MODALITY_BY_REGION.get(cur_anatomy)
    return heuristic_mod


def score_prior(
    current: Study,
    prior: Study,
) -> Tuple[float, dict]:
    """
    Score a prior study against the current study.
    Returns (score, debug_dict).

    Score interpretation:
        >= 0.5  -> relevant (True)
        <  0.5  -> not relevant (False)
    """
    debug = {}

    cur_modality = extract_modality(current.study_description) 
    pri_modality = extract_modality(prior.study_description)

    cur_anatomy = extract_anatomy(current.study_description)
    pri_anatomy = extract_anatomy(prior.study_description)

    if cur_modality is None:
        cur_modality = fallback_modality(current.study_description, cur_anatomy)

    if pri_modality is None:
        pri_modality = fallback_modality(prior.study_description, pri_anatomy)
    

    #similarity = compute_similarity(current.study_description, prior.study_description)


    debug["cur_modality"] = cur_modality
    debug["pri_modality"] = pri_modality
    debug["cur_anatomy"] = cur_anatomy
    debug["pri_anatomy"] = pri_anatomy

    #debug["similarity"] = similarity

    # --- Modality score ---
    modality_score = 0.0
    if cur_modality and pri_modality:
        if cur_modality == pri_modality:
            modality_score = 1.0
        elif pri_modality in MODALITY_COMPAT.get(cur_modality, set()):
            modality_score = 0.5
        else:
            modality_score = 0.0  # incompatible modality, strong signal
    else:
        # Unknown modality — neutral
        modality_score = 0.4

    # --- Anatomy score ---
    anatomy_score = 0.0
    if cur_anatomy and pri_anatomy:
        if cur_anatomy == pri_anatomy:
            anatomy_score = 1.0
        else:
            # Check for clinically related regions
            anatomy_score = _anatomy_relatedness(cur_anatomy, pri_anatomy)
    elif cur_anatomy is None or pri_anatomy is None:
        anatomy_score = 0.3  # can't determine, be conservative

    debug["modality_score"] = modality_score
    debug["anatomy_score"] = anatomy_score

    # --- Recency score ---
    cur_date = parse_date(current.study_date)
    pri_date = parse_date(prior.study_date)
    days = days_between(cur_date, pri_date)

    if days is None:
        recency_score = 0.5
    elif days <= 30:
        recency_score = 1.0   # same month
    elif days <= 180:
        recency_score = 0.9
    elif days <= 365:
        recency_score = 0.75
    elif days <= 730:
        recency_score = 0.6
    elif days <= 1825:
        recency_score = 0.45
    elif days <= 3650:
        recency_score = 0.3
    else:
        recency_score = 0.15  # > 10 years

    debug["days_apart"] = days
    debug["recency_score"] = recency_score

    # --- Combine scores ---
    # Anatomy is the strongest signal, modality second, recency third
    # If anatomy AND modality both mismatch strongly, it's irrelevant regardless of recency
    if anatomy_score == 0.0 and modality_score == 0.0:
        score = 0.05
    elif anatomy_score == 0.0:
        # Different body part is almost always irrelevant
        score = 0.1 + 0.1 * recency_score
    elif modality_score == 0.0:
        # Wrong modality class but right body part — sometimes relevant (e.g. CT vs MRI of same area)
        score = 0.25 + 0.2 * anatomy_score + 0.05 * recency_score
    else:
        # Weighted combination
        score = (
            0.45 * anatomy_score
            + 0.35 * modality_score
            + 0.20 * recency_score
        )

    debug["final_score"] = score
    return score, debug


def _anatomy_relatedness(a: str, b: str) -> float:
    """Return a partial relatedness score for non-identical anatomy groups."""
    # Clinically related pairs
    related_pairs = {
        frozenset(["spine_c", "brain"]): 0.4,
        frozenset(["spine_c", "face_neck"]): 0.5,
        frozenset(["spine_c", "spine_t"]): 0.3,
        frozenset(["spine_t", "spine_l"]): 0.3,
        frozenset(["spine_l", "pelvis"]): 0.4,
        frozenset(["abdomen", "pelvis"]): 0.5,
        frozenset(["chest", "abdomen"]): 0.3,
        frozenset(["chest", "vascular"]): 0.4,
        frozenset(["abdomen", "vascular"]): 0.4,
        frozenset(["pelvis", "lower_ext"]): 0.3,
        frozenset(["breast", "chest"]): 0.3,
    }
    key = frozenset([a, b])
    return related_pairs.get(key, 0.0)


RELEVANCE_THRESHOLD = 0.50


class RelevanceClassifier:
    """Classifies prior radiology studies as relevant or not relevant."""

    def __init__(self, threshold: float = RELEVANCE_THRESHOLD):
        self.threshold = threshold

    def classify_case(self, case: Case) -> List[Prediction]:
        predictions = []
        current = case.current_study

        for prior in case.prior_studies:
            score, debug = score_prior(current, prior)
            is_relevant = score >= self.threshold

            logger.debug(
                f"case={case.case_id} prior={prior.study_id} "
                f"score={score:.3f} relevant={is_relevant} debug={debug}"
            )

            predictions.append(Prediction(
                case_id=case.case_id,
                study_id=prior.study_id,
                predicted_is_relevant=is_relevant,
            ))

        return predictions
