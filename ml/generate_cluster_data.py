"""
Generate the customer dataset used by the k-means clustering stage.

Requires Python 3.10+.

    customers.csv   annual_spend, visit_frequency, segment (metadata only)

Two features per customer:
  annual_spend       thousands of dollars spent per year
  visit_frequency    store visits per month

Four natural segments are simulated, each a 2-D Gaussian blob:

    VIP               high spend, frequent visits
    Occasional        high spend, rare visits (splurges a few times a year)
    Regular           low spend, frequent visits (budget-conscious repeat visitor)
    Casual            low spend, rare visits

`segment` is written out for reference only -- k-means never sees it. It lets
the app show, after clustering, how closely the discovered groups line up
with the ones the data was generated from.

Run:  python3.10 ml/generate_cluster_data.py
"""

import os

import numpy as np
import pandas as pd

SEED = 11
POINTS_PER_SEGMENT = 45

# (label, spend_mean, spend_sd, visits_mean, visits_sd)
SEGMENTS = [
    ("VIP",        86.0, 9.0,  17.5, 2.2),
    ("Occasional", 79.0, 10.0,  4.2, 1.3),
    ("Regular",    24.0, 7.0,  16.0, 2.6),
    ("Casual",     19.0, 6.0,   3.6, 1.2),
]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def generate() -> pd.DataFrame:
    """Simulate customers as 2-D Gaussian blobs around each segment's centre."""
    rng = np.random.default_rng(SEED)
    rows = []
    for label, spend_mu, spend_sd, visit_mu, visit_sd in SEGMENTS:
        spend = rng.normal(spend_mu, spend_sd, POINTS_PER_SEGMENT)
        visits = rng.normal(visit_mu, visit_sd, POINTS_PER_SEGMENT)
        for s, v in zip(spend, visits):
            rows.append((s, v, label))

    df = pd.DataFrame(rows, columns=["annual_spend", "visit_frequency", "segment"])
    df["annual_spend"] = df["annual_spend"].clip(lower=2).round(1)
    df["visit_frequency"] = df["visit_frequency"].clip(lower=0.2).round(2)

    # Shuffle so segment order is not visible in row order -- clustering must
    # find the groups from the feature values alone.
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    """Generate the customers and write the CSV into ml/data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate()

    path = os.path.join(DATA_DIR, "customers.csv")
    df.to_csv(path, index=False)
    print(f"Generating customer dataset...")
    print(f"  wrote data/customers.csv  ({len(df)} rows)")

    print("\nSummary")
    print(df[["annual_spend", "visit_frequency"]].describe().round(2).to_string())
    print("\nSegment sizes (for reference only -- k-means never sees this column)")
    print(df["segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
