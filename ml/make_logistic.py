"""
Fit LOGISTIC regression on hours-studied -> passed and export the results for
the SwiftUI app.

Requires Python 3.10+.

Unlike linear regression there is no closed form here: setting the derivative of
the cross-entropy to zero gives equations with theta inside a sigmoid, which do
not rearrange into a formula. Gradient descent is the only solver.

The export carries one section per stage of the walkthrough:

    dataset         the binary outcomes
    linearBaseline  a least-squares line fit to the 0/1 labels, for contrast
    fit             the converged model, its boundary and confusion matrices
    curve           the fitted sigmoid, sampled for plotting
    lossSurface     cross-entropy over a grid of (w, b)
    runs            per-iteration history at several learning rates

Run:  python3.10 ml/make_logistic.py
"""

import os

import numpy as np

from common import (SEED, TEST_SIZE, accuracy, confusion_matrix, design_matrix,
                    fit_linear_closed_form, fit_logistic_gd, load_dataset,
                    log_loss, logistic_loss_surface, predict, select_frames,
                    sigmoid, split, write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SOURCE_CSV = os.path.join(HERE, "data", "study_pass.csv")
OUTPUT_JSON = os.path.join(REPO, "Models", "logistic_regression.json")

# A ladder from crawling to wildly overshooting. Cross-entropy cannot blow up to
# infinity the way squared error does -- the error term (p - y) is bounded by 1
# -- so the large rates degrade the fit rather than producing NaN.
LEARNING_RATES = [0.01, 0.05, 0.2, 0.5, 1.0, 3.0]
ITERATIONS = 3000
CURVE_SAMPLES = 120


def main() -> None:
    """Fit the model, then write the JSON export."""
    df = load_dataset(SOURCE_CSV)
    x = df["hours_studied"].to_numpy()
    y = df["passed"].to_numpy()
    print(f"Loaded {len(df)} students from data/study_pass.csv")
    print(f"  passed {int(y.sum())} / {len(y)}  ({100 * y.mean():.1f}%)")

    data = split(x, y)
    X_train, X_test = design_matrix(data["x_train"]), design_matrix(data["x_test"])
    y_train, y_test = data["y_train"], data["y_test"]
    print(f"Split: {len(y_train)} train / {len(y_test)} test "
          f"(test_size={TEST_SIZE}, seed={SEED})")

    # ---- A straight line on 0/1 labels, for contrast ----------------------
    # Least squares does not know probabilities are bounded, so its predictions
    # run below 0 and above 1.
    line_theta = fit_linear_closed_form(X_train, y_train)
    line_pred_all = predict(design_matrix(x), line_theta)
    out_of_range = int(np.sum((line_pred_all < 0) | (line_pred_all > 1)))
    print("\nA least-squares line fit to the 0/1 labels:")
    print(f"  p = {line_theta[0]:.4f} * hours + {line_theta[1]:.4f}")
    print(f"  predicts outside [0, 1] for {out_of_range} of {len(x)} students "
          f"(range {line_pred_all.min():.2f} to {line_pred_all.max():.2f})")

    # ---- Gradient descent -------------------------------------------------
    print(f"\nGradient descent, {ITERATIONS} iterations each:")
    runs = []
    best = None
    record_every = max(1, ITERATIONS // 400)
    for lr in LEARNING_RATES:
        history, diverged = fit_logistic_gd(data, lr=lr, iterations=ITERATIONS,
                                            record_every=record_every)
        last = history[-1]
        print(f"  lr={lr:<6} logLoss={last['logLoss']:7.4f} "
              f"acc={last['accuracy']:.3f} test={last['testAccuracy']:.3f} "
              f"theta={np.round(last['theta'], 4)} "
              f"boundary={last.get('boundary', float('nan')):.2f}h")
        runs.append({
            "learningRate": lr,
            "iterations": ITERATIONS,
            "diverged": diverged,
            "final": None if diverged else {
                "theta": last["theta"], "logLoss": last["logLoss"],
                "accuracy": last["accuracy"], "testAccuracy": last["testAccuracy"],
                "boundary": last.get("boundary"),
            },
            "history": select_frames(history),
        })
        if best is None or last["logLoss"] < best[1]["logLoss"]:
            best = (lr, last)

    best_lr, best_frame = best
    theta = np.array(best_frame["theta"])
    print(f"\nBest run: lr={best_lr}, log-loss {best_frame['logLoss']:.4f}")

    # ---- Scores for the converged model -----------------------------------
    p_train = sigmoid(predict(X_train, theta))
    p_test = sigmoid(predict(X_test, theta))
    pred_train = (p_train >= 0.5).astype(int)
    pred_test = (p_test >= 0.5).astype(int)
    boundary = float(-theta[1] / theta[0])
    odds_ratio = float(np.exp(theta[0]))
    print(f"  boundary at {boundary:.2f} hours; each extra hour multiplies the "
          f"odds of passing by {odds_ratio:.2f}")

    # ---- The fitted curve, sampled for drawing ----------------------------
    lo, hi = float(x.min()), float(x.max())
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, CURVE_SAMPLES)
    curve = [{"x": round(float(g), 4),
              "z": round(float(theta[0] * g + theta[1]), 6),
              "p": round(float(sigmoid(np.array([theta[0] * g + theta[1]]))[0]), 6)}
             for g in grid]

    # ---- The loss landscape over (w, b) -----------------------------------
    surface = logistic_loss_surface(X_train, y_train,
                                    w_range=(-0.15, 1.45), b_range=(-8.0, 1.0))

    test_index = set(int(i) for i in data["idx_test"])
    payload = {
        "model": "logistic",
        "title": "Logistic Regression",
        "subtitle": "Predicting whether a student passes, from hours studied",
        "library": "hand-written numpy (pandas for I/O, sklearn for the split only)",
        "dataset": {
            "name": "study_pass",
            "featureNames": ["hours_studied"],
            "axisLabels": ["Hours studied"],
            "targetLabel": "Passed",
            "classNames": ["Fail", "Pass"],
            "n": len(df),
            "mins": [float(x.min())], "maxs": [float(x.max())],
            "trainCount": len(y_train), "testCount": len(y_test),
            "positiveCount": int(y.sum()), "negativeCount": int(len(y) - y.sum()),
            "points": [{"x": [float(a)], "y": int(t), "test": i in test_index}
                       for i, (a, t) in enumerate(zip(x, y))],
        },
        "linearBaseline": {
            "theta": [round(float(v), 6) for v in line_theta],
            "outOfRangeCount": out_of_range,
            "minPrediction": round(float(line_pred_all.min()), 6),
            "maxPrediction": round(float(line_pred_all.max()), 6),
        },
        "fit": {
            "learningRate": best_lr,
            "theta": [round(float(v), 6) for v in theta],
            "logLoss": round(float(log_loss(y_train, p_train)), 6),
            "accuracy": round(accuracy(y_train, pred_train), 6),
            "testAccuracy": round(accuracy(y_test, pred_test), 6),
            "boundary": round(boundary, 6),
            "oddsRatio": round(odds_ratio, 6),
            "confusion": confusion_matrix(y_train, pred_train),
            "testConfusion": confusion_matrix(y_test, pred_test),
            # Always predicting the majority class -- the bar to beat.
            "majorityAccuracy": round(float(max(y_train.mean(), 1 - y_train.mean())), 6),
        },
        "curve": curve,
        "lossSurface": surface,
        "runs": runs,
    }

    print()
    write_json(payload, OUTPUT_JSON)


if __name__ == "__main__":
    main()
