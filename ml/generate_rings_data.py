"""
Generate the dataset used by the neural network stage.

Requires Python 3.10+.

    rings.csv   x1, x2 -> label (binary 0/1)

One class forms a disc at the centre, the other a ring around it. The shape is
the whole point: the classes are cleanly separated, but the boundary between
them is a circle, and no straight line can cut a circle out of a plane. Every
model built so far in this app draws a straight boundary, so all of them fail
here -- which is the argument for hidden layers.

The generative process:
  1. Class 0: radius ~ N(0.0, 0.28), i.e. a blob centred on the origin.
  2. Class 1: radius ~ N(1.35, 0.16), i.e. a ring at a fixed distance out.
  3. Both drawn at a uniformly random angle, so the shapes are rotationally
     symmetric and nothing about the orientation carries information.
  4. Both coordinates standardized to zero mean and unit variance.

Run:  python3.10 ml/generate_rings_data.py
"""

import os

import numpy as np
import pandas as pd

SEED = 5
POINTS_PER_CLASS = 150
INNER_RADIUS, INNER_SPREAD = 0.00, 0.28
OUTER_RADIUS, OUTER_SPREAD = 1.35, 0.16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def generate() -> pd.DataFrame:
    """Simulate a central disc surrounded by a ring, as a binary problem."""
    rng = np.random.default_rng(SEED)
    n = POINTS_PER_CLASS

    def ring(mean_radius: float, spread: float) -> np.ndarray:
        angle = rng.uniform(0, 2 * np.pi, n)
        radius = np.abs(rng.normal(mean_radius, spread, n))
        return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    points = np.vstack([ring(INNER_RADIUS, INNER_SPREAD),
                        ring(OUTER_RADIUS, OUTER_SPREAD)])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    points = (points - points.mean(axis=0)) / points.std(axis=0)

    df = pd.DataFrame({
        "x1": points[:, 0].round(4),
        "x2": points[:, 1].round(4),
        "label": labels,
    })
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    """Generate the dataset and write the CSV into ml/data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate()

    path = os.path.join(DATA_DIR, "rings.csv")
    df.to_csv(path, index=False)
    print("Generating ring dataset...")
    print(f"  wrote data/rings.csv  ({len(df)} rows)")

    print("\nSummary")
    print(df[["x1", "x2"]].describe().round(2).to_string())
    print(f"\n  class 0 (inner disc): {(df.label == 0).sum()}")
    print(f"  class 1 (outer ring): {(df.label == 1).sum()}")

    radius = np.sqrt(df.x1**2 + df.x2**2)
    print("\nDistance from centre (what makes this non-linear)")
    for label in (0, 1):
        r = radius[df.label == label]
        print(f"  class {label}: {r.min():.2f} - {r.max():.2f}  (mean {r.mean():.2f})")


if __name__ == "__main__":
    main()
