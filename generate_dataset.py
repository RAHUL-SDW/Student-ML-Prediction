"""
generate_dataset.py
Generates a realistic synthetic student dataset.
Dropout is smoothly calibrated: high attendance + high CGPA = low dropout.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 800


def generate():
    attendance = np.clip(np.random.normal(75, 15, N), 30, 100)
    cgpa       = np.clip(np.random.normal(7.2, 1.1, N), 4.0, 10.0)
    sem_gpa    = np.clip(cgpa + np.random.normal(0, 0.4, N), 4.0, 10.0)
    int_marks  = np.clip(cgpa * 9 + np.random.normal(0, 5, N), 20, 90)
    backlogs   = np.random.choice([0,1,2,3,4,5], N, p=[0.50,0.22,0.13,0.08,0.05,0.02])

    coding    = np.clip(np.random.normal(60, 20, N), 0, 100)
    comm      = np.clip(np.random.normal(62, 18, N), 0, 100)
    tech      = np.clip(np.random.normal(58, 20, N), 0, 100)
    aptitude  = np.clip(np.random.normal(65, 15, N), 0, 100)
    english   = np.clip(np.random.normal(63, 17, N), 0, 100)

    internship     = np.random.choice([0, 1], N, p=[0.45, 0.55])
    projects       = np.random.choice([0, 1, 2, 3], N, p=[0.20, 0.35, 0.30, 0.15])
    certifications = np.random.choice([0, 1, 2, 3], N, p=[0.25, 0.40, 0.25, 0.10])

    gender = np.random.choice(["Male","Female","Other"], N, p=[0.50,0.47,0.03])
    branch = np.random.choice(["CSE","ECE","ME","CE","IT"], N, p=[0.30,0.20,0.20,0.15,0.15])
    year   = np.random.choice([1,2,3,4], N, p=[0.20,0.25,0.28,0.27])

    # --- Smooth, realistic dropout logic ---
    # Each factor contributes proportionally — no hard steps
    att_effect      = -0.08 * (attendance - 75)   # per % below/above 75
    cgpa_effect     = -0.55 * (cgpa - 7.0)        # per point below/above 7
    backlog_effect  =  0.35 * backlogs
    skill_effect    = -0.008 * ((coding + tech) / 2 - 60)
    noise           = np.random.normal(0, 0.6, N)

    # Intercept tuned so ~28% dropout rate overall
    dropout_score   = att_effect + cgpa_effect + backlog_effect + skill_effect + noise - 0.5
    dropout_prob    = 1 / (1 + np.exp(-dropout_score))
    dropout_risk    = (dropout_prob > 0.5).astype(int)

    # --- Placement logic (target ~62% placement rate) ---
    place_score = (
         0.35 * (cgpa - 7.0)          # CGPA contribution (centred)
        + 0.025 * (coding - 60)        # per point above/below 60
        + 0.015 * (comm - 60)
        + 0.015 * (tech - 60)
        + 0.70  * internship           # big boost for internship
        + 0.20  * projects
        + 0.12  * certifications
        + 0.010 * (aptitude - 65)
        - 0.20  * backlogs
        + np.random.normal(0, 0.8, N)
        + 0.2                          # slight positive bias
    )
    placement = (1 / (1 + np.exp(-place_score)) > 0.5).astype(int)

    df = pd.DataFrame({
        "student_id": [f"STU{i:04d}" for i in range(1, N+1)],
        "gender": gender, "branch": branch, "year": year,
        "cgpa": cgpa.round(2),
        "attendance": attendance.round(1),
        "sem_gpa": sem_gpa.round(2),
        "internal_marks": int_marks.round(1),
        "backlogs": backlogs,
        "coding_skill": coding.round(1),
        "communication_skill": comm.round(1),
        "technical_skill": tech.round(1),
        "aptitude_score": aptitude.round(1),
        "english_proficiency": english.round(1),
        "internship": internship,
        "projects": projects,
        "certifications": certifications,
        "dropout_risk": dropout_risk,
        "placement": placement,
    })

    # 5% random missing in less-critical columns
    for col in ["sem_gpa","internal_marks","english_proficiency","certifications"]:
        mask = np.random.rand(N) < 0.05
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/students_raw.csv", index=False)
    print(f"Saved {df.shape[0]} rows × {df.shape[1]} cols")
    print("Dropout rate:", df["dropout_risk"].mean().round(3))
    print("Placement rate:", df["placement"].mean().round(3))
