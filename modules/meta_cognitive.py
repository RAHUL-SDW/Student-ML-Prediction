"""
modules/meta_cognitive.py
Meta-Cognitive Evaluation Module — evaluates prediction reliability.

For every prediction, it computes:
  - Prediction Confidence   : model's probability margin
  - Data Completeness       : fraction of expected features present
  - Academic Consistency    : derived from CGPA/attendance/backlogs signals
  - Skill Readiness         : derived from skill features
  - Feature Reliability     : variance-based trust of feature values
  - Final Reliability Score : weighted aggregate
"""

import numpy as np

EXPECTED_FEATURES = [
    "cgpa", "attendance", "sem_gpa", "internal_marks", "backlogs",
    "coding_skill", "communication_skill", "technical_skill",
    "aptitude_score", "english_proficiency",
    "internship", "projects", "certifications",
]


def _clamp(v, lo=0.0, hi=100.0):
    return float(np.clip(v, lo, hi))


def compute_prediction_confidence(proba: float) -> float:
    """
    Confidence based on how far probability is from 0.5 decision boundary.
    proba: raw model probability for the positive class.
    """
    margin = abs(proba - 0.5) * 2          # 0 at boundary, 1 at extremes
    confidence = 50 + margin * 50           # maps to [50, 100]
    return _clamp(confidence)


def compute_data_completeness(input_dict: dict) -> float:
    present = sum(
        1 for f in EXPECTED_FEATURES
        if input_dict.get(f) not in (None, "", float("nan"))
    )
    return _clamp(present / len(EXPECTED_FEATURES) * 100)


def compute_academic_consistency(input_dict: dict) -> float:
    cgpa = float(input_dict.get("cgpa") or 7)
    attendance = float(input_dict.get("attendance") or 75)
    backlogs = float(input_dict.get("backlogs") or 0)
    sem_gpa = float(input_dict.get("sem_gpa") or cgpa)

    cgpa_score = (cgpa / 10) * 100
    att_score = attendance
    backlog_penalty = min(backlogs * 10, 40)
    consistency_gap = abs(cgpa - sem_gpa) * 10

    score = (cgpa_score * 0.4 + att_score * 0.4) - backlog_penalty - consistency_gap
    return _clamp(score)


def compute_skill_readiness(input_dict: dict) -> float:
    skills = [
        input_dict.get("coding_skill"),
        input_dict.get("communication_skill"),
        input_dict.get("technical_skill"),
        input_dict.get("aptitude_score"),
        input_dict.get("english_proficiency"),
    ]
    valid = [float(s) for s in skills if s not in (None, "", float("nan"))]
    if not valid:
        return 50.0

    base = np.mean(valid)
    # Bonus for experience
    internship_bonus = float(input_dict.get("internship") or 0) * 5
    project_bonus = float(input_dict.get("projects") or 0) * 3
    cert_bonus = float(input_dict.get("certifications") or 0) * 2
    return _clamp(base + internship_bonus + project_bonus + cert_bonus)


def compute_feature_reliability(input_dict: dict) -> float:
    """
    Penalises extreme/unlikely feature values that reduce trust in the input.
    """
    penalty = 0.0

    cgpa = float(input_dict.get("cgpa") or 7)
    if cgpa < 4 or cgpa > 10:
        penalty += 15

    attendance = float(input_dict.get("attendance") or 75)
    if attendance < 20 or attendance > 100:
        penalty += 10

    for skill_key in ["coding_skill", "communication_skill", "technical_skill"]:
        v = input_dict.get(skill_key)
        if v is not None:
            v = float(v)
            if v < 0 or v > 100:
                penalty += 5

    return _clamp(100 - penalty)


def evaluate_prediction(
    input_dict: dict,
    dropout_proba: float,
    placement_proba: float,
) -> dict:
    """
    Main function: run all five sub-evaluators and compute the final score.
    Returns a dict of all scores (0–100 floats).
    """
    confidence = (
        compute_prediction_confidence(dropout_proba) * 0.5
        + compute_prediction_confidence(placement_proba) * 0.5
    )
    completeness = compute_data_completeness(input_dict)
    academic = compute_academic_consistency(input_dict)
    skill = compute_skill_readiness(input_dict)
    reliability = compute_feature_reliability(input_dict)

    # Weighted final reliability score
    final = (
        confidence * 0.30
        + completeness * 0.20
        + academic * 0.20
        + skill * 0.15
        + reliability * 0.15
    )

    return {
        "prediction_confidence": round(confidence, 1),
        "data_completeness": round(completeness, 1),
        "academic_consistency": round(academic, 1),
        "skill_readiness": round(skill, 1),
        "feature_reliability": round(reliability, 1),
        "final_reliability_score": round(final, 1),
    }


def reliability_label(score: float) -> str:
    if score >= 85:
        return "High"
    elif score >= 65:
        return "Medium"
    else:
        return "Low"
