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
    empiricalRate   the observed pass rate in each band of hours
    thresholdSweep  scores at every decision cut, not just 0.5
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

    # ---- The observed pass rate in each slice of hours ---------------------
    # Grouping the raw 0/1 outcomes and averaging them gives the proportion who
    # passed in each band, which already traces an S even before anything is
    # fitted.
    edges = np.linspace(lo, hi, 11)
    empirical = []
    for start, end in zip(edges[:-1], edges[1:]):
        inside = (x >= start) & (x <= end if end == edges[-1] else x < end)
        count = int(inside.sum())
        if count == 0:
            continue
        empirical.append({
            "binStart": round(float(start), 4),
            "binEnd": round(float(end), 4),
            "center": round(float((start + end) / 2), 4),
            "count": count,
            "passed": int(y[inside].sum()),
            "rate": round(float(y[inside].mean()), 6),
        })
    print("\nObserved pass rate by hours studied:")
    for band in empirical:
        bar = "#" * int(round(band["rate"] * 24))
        print(f"  {band['binStart']:5.2f}-{band['binEnd']:5.2f}h  "
              f"{band['passed']:3d}/{band['count']:<3d} = {band['rate']:.2f}  {bar}")

    # ---- Scores at every decision threshold --------------------------------
    # 0.5 is only a default. Moving the cut changes which mistakes are made:
    # a lower threshold catches more passes but wrongly flags more failures.
    sweep = []
    for cut in np.round(np.arange(0.05, 0.96, 0.05), 2):
        train_at = (p_train >= cut).astype(int)
        test_at = (p_test >= cut).astype(int)
        tn, fp = confusion_matrix(y_train, train_at)[0]
        fn, tp = confusion_matrix(y_train, train_at)[1]
        sweep.append({
            "threshold": float(cut),
            "accuracy": round(accuracy(y_train, train_at), 6),
            "testAccuracy": round(accuracy(y_test, test_at), 6),
            "confusion": confusion_matrix(y_train, train_at),
            "testConfusion": confusion_matrix(y_test, test_at),
            # Of those flagged as passing, how many did; of those who passed,
            # how many were caught.
            "precision": round(float(tp / (tp + fp)), 6) if tp + fp > 0 else 0.0,
            "recall": round(float(tp / (tp + fn)), 6) if tp + fn > 0 else 0.0,
        })
    best_cut = max(sweep, key=lambda s: s["accuracy"])
    print(f"\nThreshold sweep: best training accuracy {best_cut['accuracy']:.3f} "
          f"at p >= {best_cut['threshold']}")

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
        "empiricalRate": empirical,
        "thresholdSweep": sweep,
        "lossSurface": surface,
        "runs": runs,
    }

    print()
    write_json(payload, OUTPUT_JSON)


if __name__ == "__main__":
    main()
