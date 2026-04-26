"""
Unit tests for the relevance classifier.
Run with: pytest tests/
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classifier import (
    extract_modality,
    extract_anatomy,
    score_prior,
    RelevanceClassifier,
)
from src.models import Study, Case


def make_study(study_id, description, date="2023-01-01"):
    return Study(study_id=study_id, study_description=description, study_date=date)


def make_case(case_id, current, priors):
    return Case(
        case_id=case_id,
        patient_id="P001",
        current_study=current,
        prior_studies=priors,
    )


class TestModalityExtraction:
    def test_mri(self):
        assert extract_modality("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST") == "MRI"

    def test_ct(self):
        assert extract_modality("CT HEAD WITHOUT CNTRST") == "CT"

    def test_xray(self):
        assert extract_modality("CHEST X-RAY AP") == "XRAY"

    def test_ultrasound(self):
        assert extract_modality("ULTRASOUND ABDOMEN") == "US"

    def test_pet(self):
        assert extract_modality("PET CT WHOLE BODY") == "NM"

    def test_mammography(self):
        assert extract_modality("MAMMOGRAM BILATERAL") == "MAMMO"

    def test_unknown(self):
        assert extract_modality("UNKNOWN STUDY TYPE") is None


class TestAnatomyExtraction:
    def test_brain(self):
        assert extract_anatomy("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST") == "brain"

    def test_chest(self):
        assert extract_anatomy("CT CHEST WITH CONTRAST") == "chest"

    def test_lumbar(self):
        assert extract_anatomy("MRI LUMBAR SPINE WITHOUT CONTRAST") == "spine_l"

    def test_abdomen(self):
        assert extract_anatomy("CT ABDOMEN AND PELVIS") == "abdomen"


class TestScoring:
    def test_exact_match_relevant(self):
        current = make_study("C1", "MRI BRAIN WITHOUT CONTRAST", "2023-06-01")
        prior = make_study("P1", "MRI BRAIN WITHOUT CONTRAST", "2022-06-01")
        score, debug = score_prior(current, prior)
        assert score >= 0.5, f"Expected relevant, got score={score}"

    def test_different_anatomy_irrelevant(self):
        current = make_study("C1", "MRI BRAIN WITHOUT CONTRAST", "2023-06-01")
        prior = make_study("P1", "X-RAY CHEST AP", "2022-06-01")
        score, debug = score_prior(current, prior)
        assert score < 0.5, f"Expected irrelevant, got score={score}"

    def test_same_anatomy_different_modality_relevant(self):
        current = make_study("C1", "MRI BRAIN WITHOUT CONTRAST", "2023-06-01")
        prior = make_study("P1", "CT HEAD WITHOUT CONTRAST", "2022-06-01")
        score, debug = score_prior(current, prior)
        assert score >= 0.5, f"Expected relevant, got score={score}"

    def test_very_old_study_lower_score(self):
        current = make_study("C1", "MRI BRAIN", "2023-06-01")
        recent_prior = make_study("P1", "MRI BRAIN", "2022-06-01")
        old_prior = make_study("P2", "MRI BRAIN", "2000-01-01")
        score_recent, _ = score_prior(current, recent_prior)
        score_old, _ = score_prior(current, old_prior)
        assert score_recent > score_old, "Recent prior should score higher"

    def test_chest_ct_chest_xray_relevant(self):
        current = make_study("C1", "CT CHEST WITH CONTRAST", "2023-06-01")
        prior = make_study("P1", "CHEST X-RAY AP", "2022-01-01")
        score, _ = score_prior(current, prior)
        assert score >= 0.4  # chest xray is related to chest CT


class TestClassifier:
    def setup_method(self):
        self.clf = RelevanceClassifier()

    def test_returns_predictions_for_all_priors(self):
        case = make_case(
            "C1",
            make_study("CS1", "MRI BRAIN WITHOUT CONTRAST", "2023-06-01"),
            [
                make_study("P1", "MRI BRAIN WITHOUT CONTRAST", "2022-01-01"),
                make_study("P2", "X-RAY CHEST AP", "2021-01-01"),
            ],
        )
        preds = self.clf.classify_case(case)
        assert len(preds) == 2
        study_ids = {p.study_id for p in preds}
        assert study_ids == {"P1", "P2"}

    def test_brain_brain_relevant(self):
        case = make_case(
            "C2",
            make_study("CS1", "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST", "2023-06-01"),
            [make_study("P1", "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST", "2020-03-08")],
        )
        preds = self.clf.classify_case(case)
        assert preds[0].predicted_is_relevant is True

    def test_brain_chest_irrelevant(self):
        case = make_case(
            "C3",
            make_study("CS1", "MRI BRAIN WITHOUT CONTRAST", "2023-06-01"),
            [make_study("P1", "CT CHEST WITH CONTRAST", "2022-01-01")],
        )
        preds = self.clf.classify_case(case)
        assert preds[0].predicted_is_relevant is False

    def test_example_from_spec(self):
        """Test the exact example from the challenge spec."""
        case = make_case(
            "1001016",
            make_study("3100042", "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST", "2026-03-08"),
            [
                make_study("2453245", "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST", "2020-03-08"),
                make_study("992654", "CT HEAD WITHOUT CNTRST", "2021-03-08"),
            ],
        )
        preds = self.clf.classify_case(case)
        pred_map = {p.study_id: p.predicted_is_relevant for p in preds}
        # Both should be relevant — same/related anatomy+modality
        assert pred_map["2453245"] is True, "Same MRI brain should be relevant"
        assert pred_map["992654"] is True, "CT head when current is MRI brain should be relevant"
