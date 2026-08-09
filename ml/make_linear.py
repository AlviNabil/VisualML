"""
Fit linear regression with both solvers and export the results for the
SwiftUI app.

Requires Python 3.10+.

Two models are built from the same cohort and the same target column:

    linear_regression.json        score ~ hours                 (a line)
    linear_regression_multi.json  score ~ hours + sleep         (a plane)

    solver 1  closed form      -- the normal equations, exact, one step
    solver 2  gradient descent -- iterative, swept across several learning rates

Each JSON holds the dataset, both solutions, the per-iteration history of every
gradient-descent run, and the residuals of the closed-form fit.

Run:  python3.10 ml/make_linear.py
"""

import math
import os

import numpy as np

from common import (SEED, TEST_SIZE, design_matrix, fit_linear_closed_form,
                    fit_linear_gd, load_dataset, loss_quadratic, mse, predict,
                    r_squared, rmse, select_frames, split, stability_limit,
                    write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODELS_DIR = os.path.join(REPO, "Models")

# Learning rates are expressed as fractions of the dataset's stability limit
# 2/lambda_max, so the same ladder is meaningful for any number of features.
# The last entry sits above the limit and diverges.
LR_FRACTIONS = [0.02, 0.10, 0.30, 0.60, 0.90, 1.05]


def verify_against_sklearn(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
    """Compare the closed-form parameters with sklearn's LinearRegression.

    Prints OK when both agree to within 1e-8. sklearn is used as a test oracle
    only; no exported value comes from it.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return
    ref = LinearRegression().fit(X[:, :-1], y)
    ref_theta = np.append(ref.coef_, ref.intercept_)
    gap = float(np.max(np.abs(theta - ref_theta)))
    print(f"  cross-check vs sklearn LinearRegression: "
          f"{'OK' if gap < 1e-8 else 'MISMATCH'} (max diff {gap:.2e})")


def build(source_csv: str, features: list[str], target: str, output_json: str,
          title: str, subtitle: str, axis_labels: list[str], target_label: str,
          iterations: int) -> None:
    """Fit both solvers on the given feature columns and write the JSON export."""
    path = os.path.join(HERE, "data", source_csv)
    df = load_dataset(path)
    x = df[features].to_numpy()
    y = df[target].to_numpy()

    print(f"\n{'=' * 70}\n{title}  --  {' + '.join(features)} -> {target}\n{'=' * 70}")
    print(f"Loaded {len(df)} rows from data/{source_csv}")

    data = split(x, y)
    X_train, X_test = design_matrix(data["x_train"]), design_matrix(data["x_test"])
    y_train, y_test = data["y_train"], data["y_test"]
    print(f"Split: {len(y_train)} train / {len(y_test)} test "
          f"(test_size={TEST_SIZE}, seed={SEED})")

    # ---- Solver 1: closed form -------------------------------------------
    theta_star = fit_linear_closed_form(X_train, y_train)
    pred_train, pred_test = predict(X_train, theta_star), predict(X_test, theta_star)

    terms = " + ".join(f"{theta_star[i]:.4f}*{f}" for i, f in enumerate(features))
    print("\nSOLVER 1 -- closed form (normal equations):")
    print(f"  {target} = {terms} + {theta_star[-1]:.4f}")
    print(f"  train  MSE {mse(y_train, pred_train):8.3f}  "
          f"RMSE {rmse(y_train, pred_train):6.3f}  R^2 {r_squared(y_train, pred_train):.4f}")
    print(f"  test   MSE {mse(y_test, pred_test):8.3f}  "
          f"RMSE {rmse(y_test, pred_test):6.3f}  R^2 {r_squared(y_test, pred_test):.4f}")
    verify_against_sklearn(X_train, y_train, theta_star)

    # ---- Solver 2: gradient descent --------------------------------------
    limit = stability_limit(X_train)
    record_every = max(1, iterations // 400)
    print(f"\nSOLVER 2 -- batch gradient descent, {iterations} iterations each")
    print(f"  stability limit 2/lambda_max = {limit:.6f}")

    runs = []
    for fraction in LR_FRACTIONS:
        lr = limit * fraction
        history, diverged = fit_linear_gd(data, lr=lr, iterations=iterations,
                                          record_every=record_every)
        last = history[-1]
        gap = float(np.max(np.abs(np.array(last["theta"]) - theta_star)))
        status = (f"DIVERGED after {len(history)} steps" if diverged else
                  f"theta={np.round(last['theta'], 3)} "
                  f"MSE={last['mse']:8.3f} R^2={last['r2']:.4f} "
                  f"(off closed form by {gap:.3f})")
        print(f"  lr={lr:<10.6f} ({fraction:>4.2f} x limit)  {status}")

        runs.append({
            "learningRate": round(lr, 8),
            "fractionOfLimit": fraction,
            "iterations": iterations,
            "diverged": diverged,
            "final": None if diverged else {
                "theta": last["theta"],
                "mse": last["mse"], "r2": last["r2"],
                "rmse": round(math.sqrt(last["mse"]), 6),
                "testMse": last["testMse"], "testR2": last["testR2"],
            },
            "history": select_frames(history),
        })

    # Residuals of the closed-form fit, for every row in the file.
    residuals = (y - predict(design_matrix(x), theta_star)).round(3).tolist()

    mean_prediction = float(np.mean(y_train))
    test_set = set(int(i) for i in data["idx_test"])
    payload = {
        "model": "linear",
        "featureCount": len(features),
        "title": title,
        "subtitle": subtitle,
        "library": "hand-written numpy (pandas for I/O, sklearn for the split only)",
        "dataset": {
            "name": source_csv.replace(".csv", ""),
            "featureNames": features,
            "axisLabels": axis_labels,
            "targetLabel": target_label,
            "n": len(df),
            "mins": [float(v) for v in x.min(axis=0)],
            "maxs": [float(v) for v in x.max(axis=0)],
            "targetMin": float(y.min()), "targetMax": float(y.max()),
            "trainCount": len(y_train), "testCount": len(y_test),
            "points": [{"x": [float(v) for v in row], "y": float(t), "test": i in test_set}
                       for i, (row, t) in enumerate(zip(x, y))],
        },
        "closedForm": {
            "theta": [round(float(t), 6) for t in theta_star],
            "mse": round(mse(y_train, pred_train), 6),
            "rmse": round(rmse(y_train, pred_train), 6),
            "r2": round(r_squared(y_train, pred_train), 6),
            "testMse": round(mse(y_test, pred_test), 6),
            "testR2": round(r_squared(y_test, pred_test), 6),
        },
        # Metrics for always predicting the training mean.
        "baseline": {
            "prediction": round(mean_prediction, 6),
            "mse": round(mse(y_train, np.full_like(y_train, mean_prediction)), 6),
        },
        # Statistics that let the app evaluate the loss surface and its slope
        # at any theta, for the gradient-descent explainer.
        "lossQuadratic": loss_quadratic(X_train, y_train),
        "residuals": residuals,
        "runs": runs,
    }
    write_json(payload, os.path.join(MODELS_DIR, output_json))


def main() -> None:
    """Build the single-feature and two-feature exports."""
    build(source_csv="study_score.csv",
          features=["hours_studied"],
          target="exam_score",
          output_json="linear_regression.json",
          title="Linear Regression",
          subtitle="Predicting an exam score from hours studied",
          axis_labels=["Hours studied"],
          target_label="Exam score",
          iterations=2000)

    build(source_csv="study_score_multi.csv",
          features=["hours_studied", "sleep_hours"],
          target="exam_score",
          output_json="linear_regression_multi.json",
          title="Multiple Linear Regression",
          subtitle="Fitting a plane through hours studied and sleep",
          axis_labels=["Hours studied", "Sleep hours"],
          target_label="Exam score",
          # The two-feature problem is far more ill-conditioned (condition
          # number ~5300 vs ~330), so it needs many more steps to settle.
          iterations=50000)


if __name__ == "__main__":
    main()
