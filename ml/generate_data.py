"""
Generate the study-habit dataset used by the linear regression stage.

Requires Python 3.10+.

    study_score.csv        hours         -> score   (continuous 0..100)
    study_score_multi.csv  hours, sleep  -> score   (continuous 0..100)

The score column is identical in both files, so the single-feature model and
the two-feature model predict the same target from different amounts of
information.

The generative process:
  1. hours ~ Beta(2.2, 2.4) rescaled to [0.5, 12].
  2. sleep ~ Normal(7, 1.1) clipped to [3.5, 10].
  3. A curved contribution from each:
         hours:  7.6*h - 0.30*h^2       (diminishing returns)
         sleep:  3.4*s - 0.28*(s-7.5)^2 (a penalty for too little or too much)
  4. Gaussian noise whose spread shrinks as hours grow (9.5 - 0.45*h, floored
     at 3.0), making the data heteroscedastic.
  5. Roughly 5% of rows receive an extra +/- shock, producing outliers.
  6. Scores are clipped to [0, 100] and rounded to one decimal.

Run:  python3.10 ml/generate_data.py
"""

import os

import numpy as np
import pandas as pd

SEED = 7
N_STUDENTS = 200

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def generate() -> pd.DataFrame:
    """Simulate N_STUDENTS rows of hours_studied, sleep_hours and exam_score."""
    rng = np.random.default_rng(SEED)

    hours = 0.5 + 11.5 * rng.beta(a=2.2, b=2.4, size=N_STUDENTS)
    sleep = np.clip(rng.normal(7.0, 1.1, size=N_STUDENTS), 3.5, 10.0)

    trend = (2.0
             + 7.6 * hours - 0.30 * hours**2
             + 3.4 * sleep - 0.28 * (sleep - 7.5) ** 2)

    noise_sd = np.maximum(9.5 - 0.45 * hours, 3.0)
    score = trend + rng.normal(0.0, noise_sd)

    n_outliers = max(1, int(0.05 * N_STUDENTS))
    idx = rng.choice(N_STUDENTS, size=n_outliers, replace=False)
    score[idx] += rng.normal(0.0, 15.0, size=n_outliers)

    return pd.DataFrame({
        "hours_studied": hours.round(2),
        "sleep_hours": sleep.round(2),
        "exam_score": np.clip(score, 0.0, 100.0).round(1),
    })


def main() -> None:
    """Generate the cohort and write the CSVs into ml/data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate()

    print("Generating study-habit datasets...")
    df[["hours_studied", "exam_score"]].to_csv(
        os.path.join(DATA_DIR, "study_score.csv"), index=False)
    df[["hours_studied", "sleep_hours", "exam_score"]].to_csv(
        os.path.join(DATA_DIR, "study_score_multi.csv"), index=False)
    print(f"  wrote data/study_score.csv        ({len(df)} rows)")
    print(f"  wrote data/study_score_multi.csv  ({len(df)} rows)")

    print("\nSummary")
    print(df[["hours_studied", "sleep_hours", "exam_score"]]
          .describe().round(2).to_string())

    corr = df[["hours_studied", "sleep_hours", "exam_score"]].corr().round(3)
    print("\nCorrelations")
    print(corr.to_string())


if __name__ == "__main__":
    main()
