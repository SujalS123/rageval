# tests/test_noise_sensitivity.py

import pytest
from unittest.mock import MagicMock
from rageval.metrics.noise_sensitivity import NoiseSensitivity
from rageval.core.sample import RAGSample

NOISE_CORPUS = [
    "The French Revolution began in 1789.",
    "Mount Everest is the tallest mountain on Earth.",
    "The speed of light is approximately 299,792 km/s.",
]

# Faithfulness calls complete_json twice per run: extraction then verification.
# NoiseSensitivity runs Faithfulness twice, so 4 complete_json calls total.

def make_mock_judge(clean_extraction, clean_verification, noisy_extraction, noisy_verification):
    judge = MagicMock()
    judge.complete_json.side_effect = [
        clean_extraction,
        clean_verification,
        noisy_extraction,
        noisy_verification,
    ]
    return judge


def make_sample():
    return RAGSample(
        query="What is the boiling point of water?",
        retrieved_docs=["Water boils at 100 degrees Celsius at sea level."],
        answer="Water boils at 100 degrees Celsius.",
    )


def test_robust_pipeline_scores_near_1():
    """Pipeline unaffected by noise — clean and noisy faithfulness both 1.0, score = 1.0."""
    full_support = {
        "verifications": [
            {"claim": "Water boils at 100 degrees Celsius.", "supported": True, "reason": "Context states this."}
        ]
    }
    judge = make_mock_judge(
        clean_extraction={"claims": ["Water boils at 100 degrees Celsius."]},
        clean_verification=full_support,
        noisy_extraction={"claims": ["Water boils at 100 degrees Celsius."]},
        noisy_verification=full_support,
    )

    metric = NoiseSensitivity(judge=judge, noise_corpus=NOISE_CORPUS, n_noise=1, threshold=0.8)
    result = metric.score(make_sample())

    assert result.score == 1.0
    assert result.passed is True


def test_fragile_pipeline_scores_low():
    """Clean faithfulness 1.0, noisy faithfulness 0.0 — degradation 1.0, score 0.0."""
    judge = make_mock_judge(
        clean_extraction={"claims": ["Water boils at 100 degrees Celsius."]},
        clean_verification={
            "verifications": [
                {"claim": "Water boils at 100 degrees Celsius.", "supported": True, "reason": "Context confirms."}
            ]
        },
        noisy_extraction={"claims": ["Water boils at 100 degrees Celsius."]},
        noisy_verification={
            "verifications": [
                {"claim": "Water boils at 100 degrees Celsius.", "supported": False, "reason": "Lost in noise."}
            ]
        },
    )

    metric = NoiseSensitivity(judge=judge, noise_corpus=NOISE_CORPUS, n_noise=1, threshold=0.8)
    result = metric.score(make_sample())

    assert result.score == 0.0
    assert result.passed is False


def test_evidence_contains_noise_docs():
    """Evidence must include the specific noise documents that were injected."""
    full_support = {
        "verifications": [
            {"claim": "Water boils at 100 degrees Celsius.", "supported": True, "reason": "Context confirms."}
        ]
    }
    judge = make_mock_judge(
        clean_extraction={"claims": ["Water boils at 100 degrees Celsius."]},
        clean_verification=full_support,
        noisy_extraction={"claims": ["Water boils at 100 degrees Celsius."]},
        noisy_verification=full_support,
    )

    # Use a single-item corpus so we know exactly which doc gets injected
    fixed_noise = ["This is irrelevant noise about cooking recipes."]
    metric = NoiseSensitivity(judge=judge, noise_corpus=fixed_noise, n_noise=1, threshold=0.8)
    result = metric.score(make_sample())

    assert any("cooking recipes" in e for e in result.evidence)
    assert any("Noise doc injected" in e for e in result.evidence)


def test_degradation_correctly_computed():
    """Degradation = clean_score - noisy_score when positive, score = 1 - degradation."""
    # clean: 1 of 2 supported = 0.5, noisy: 0 of 2 supported = 0.0 -> degradation 0.5 -> score 0.5
    judge = make_mock_judge(
        clean_extraction={"claims": ["Claim A.", "Claim B."]},
        clean_verification={
            "verifications": [
                {"claim": "Claim A.", "supported": True, "reason": "Found."},
                {"claim": "Claim B.", "supported": False, "reason": "Not found."},
            ]
        },
        noisy_extraction={"claims": ["Claim A.", "Claim B."]},
        noisy_verification={
            "verifications": [
                {"claim": "Claim A.", "supported": False, "reason": "Confused by noise."},
                {"claim": "Claim B.", "supported": False, "reason": "Not found."},
            ]
        },
    )

    metric = NoiseSensitivity(judge=judge, noise_corpus=NOISE_CORPUS, n_noise=1, threshold=0.8)
    result = metric.score(make_sample())

    # clean=0.5, noisy=0.0, degradation=0.5, score=0.5
    assert result.score == pytest.approx(0.5, abs=0.01)
    assert any("0.500" in e or "0.5" in e for e in result.evidence if "Degradation" in e)
