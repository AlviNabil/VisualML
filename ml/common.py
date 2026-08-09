"""
Regression math shared by make_linear.py and make_logistic.py.

Requires Python 3.10+.

The bias is folded into the parameter vector: every input row is augmented with
a constant 1, so

    X     = [x, 1]        shape (n, 2)
    theta = [w, b]        shape (2,)
    y_hat = X @ theta     = w*x + b

so there is a single parameter vector to solve for, differentiate and update.

Division of labour:
  * pandas       -- reading the CSVs
  * scikit-learn -- the train/test split only
  * numpy + this file -- all of the mathematics
"""

import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 7
TEST_SIZE = 0.25


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> pd.DataFrame:
    """Read a two-column CSV (feature, target) with pandas."""
    return pd.read_csv(path)


def split(x: np.ndarray, y: np.ndarray) -> dict:
    """Split x and y into train and test sets.

    Returns a dict holding the two halves plus the original row indices of
    each, so callers can tell which rows were held out.
    """
    idx = np.arange(len(x))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, idx, test_size=TEST_SIZE, random_state=SEED)
    return {
        "x_train": x_train, "y_train": y_train, "idx_train": idx_train,
        "x_test": x_test, "y_test": y_test, "idx_test": idx_test,
    }


def design_matrix(x: np.ndarray) -> np.ndarray:
    """Augment one or more feature columns with a trailing column of ones.

    Accepts x of shape (n,) or (n, d) and returns X of shape (n, d+1), so that
    X @ theta evaluates w1*x1 + ... + wd*xd + b with theta = [w1..wd, b].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([x, np.ones(len(x))])


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the linear score   X @ theta."""
    return X @ theta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean Squared Error:   J = (1/n) * sum( (y_hat - y)^2 )"""
    return float(np.mean((y_hat - y) ** 2))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root Mean Squared Error, in the same units as y."""
    return math.sqrt(mse(y, y_hat))


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination:   R^2 = 1 - SS_res / SS_tot

        SS_res = sum( (y - y_hat)^2 )
        SS_tot = sum( (y - mean(y))^2 )

    Returns the fraction of the variance in y the model accounts for: 1 is a
    perfect fit, 0 matches always predicting the mean, negative is worse.
    """
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of labels predicted correctly."""
    return float(np.mean(y == predicted))


def confusion_matrix(y: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    """2x2 label counts indexed as [actual][predicted]."""
    return [[int(np.sum((y == a) & (predicted == p))) for p in (0, 1)]
            for a in (0, 1)]


# ---------------------------------------------------------------------------
# Linear regression -- solver 1: closed form
# ---------------------------------------------------------------------------


def fit_linear_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares exactly via the normal equations.

    Setting the gradient of the MSE to zero gives

        X^T X theta = X^T y

    which is solved directly with np.linalg.solve rather than by forming the
    inverse. With X = [x, 1] the returned theta is [w, b].
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


# ---------------------------------------------------------------------------
# Linear regression -- solver 2: gradient descent
# ---------------------------------------------------------------------------


def stability_limit(X: np.ndarray) -> float:
    """Return the learning rate at which batch gradient descent starts to diverge.

    For a squared-error loss the curvature is constant and given by the Hessian

        H = (2/n) * X^T X

    Convergence holds while  lr < 2 / lambda_max(H) , which is the value
    returned here.
    """
    hessian = 2.0 / X.shape[0] * (X.T @ X)
    return float(2.0 / np.max(np.linalg.eigvalsh(hessian)))


def fit_linear_gd(data: dict, lr: float, iterations: int,
                  record_every: int = 1) -> tuple[list[dict], bool]:
    """Fit y_hat = X @ theta by batch gradient descent.

    Loss:      J(theta) = (1/n) * ||X theta - y||^2
    Gradient:  grad J   = (2/n) * X^T (X theta - y)
    Update:    theta   <- theta - lr * grad J

    Every step uses all training rows. Starts from theta = 0 and stops early if
    the loss becomes non-finite or exceeds 1e12.

    A snapshot is taken for the first 60 iterations, then every `record_every`
    iterations, and always on the final one.

    Returns (history, diverged), where history holds one dict per snapshot with
    the parameters and the train/test metrics at that point.
    """
    X, y = design_matrix(data["x_train"]), data["y_train"]
    X_test, y_test = design_matrix(data["x_test"]), data["y_test"]
    n = X.shape[0]
    theta = np.zeros(X.shape[1])
    history: list[dict] = []

    for i in range(iterations + 1):
        y_hat = predict(X, theta)
        residual = y_hat - y
        loss = float(np.mean(residual**2))

        if not math.isfinite(loss) or loss > 1e12:
            return history, True

        if i < 60 or i % record_every == 0 or i == iterations:
            y_hat_test = predict(X_test, theta)
            history.append({
                "iter": i,
                "theta": [round(float(t), 6) for t in theta],
                "mse": round(loss, 6),
                "r2": round(r_squared(y, y_hat), 6),
                "testMse": round(mse(y_test, y_hat_test), 6),
                "testR2": round(r_squared(y_test, y_hat_test), 6),
            })

        if i == iterations:
            break

        theta = theta - lr * (2.0 / n) * (X.T @ residual)

    return history, False


# ---------------------------------------------------------------------------
# Logistic regression -- gradient descent
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Evaluate   sigma(z) = 1 / (1 + e^-z),  mapping the reals onto (0, 1).

    Uses a branch on the sign of z so exp() never overflows.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy:

        J = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )

    Probabilities are clipped away from 0 and 1 to keep log() finite.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic_gd(data: dict, lr: float, iterations: int) -> tuple[list[dict], bool]:
    """Fit P(y=1) = sigma(X @ theta) by batch gradient descent, recording every step.

    Model:     z = X theta ,  p = sigma(z)
    Loss:      J(theta) = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )
    Gradient:  grad J   = (1/n) * X^T (p - y)
    Update:    theta   <- theta - lr * grad J

    Each recorded step also stores `boundary`, the x where p = 0.5, i.e. the
    solution of z = 0.

    Returns (history, diverged).
    """
    X, y = design_matrix(data["x_train"]), data["y_train"]
    X_test, y_test = design_matrix(data["x_test"]), data["y_test"]
    n = X.shape[0]
    theta = np.zeros(X.shape[1])
    history: list[dict] = []

    for i in range(iterations + 1):
        p = sigmoid(predict(X, theta))
        loss = log_loss(y, p)

        if not math.isfinite(loss) or loss > 1e12:
            return history, True

        p_test = sigmoid(predict(X_test, theta))
        frame = {
            "iter": i,
            "theta": [round(float(t), 6) for t in theta],
            "logLoss": round(loss, 6),
            "accuracy": round(accuracy(y, (p >= 0.5).astype(int)), 6),
            "testAccuracy": round(accuracy(y_test, (p_test >= 0.5).astype(int)), 6),
        }
        if theta.size == 2 and abs(theta[0]) > 1e-9:
            frame["boundary"] = round(float(-theta[1] / theta[0]), 6)
        history.append(frame)

        if i == iterations:
            break

        theta = theta - lr * (1.0 / n) * (X.T @ (p - y))

    return history, False


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def select_frames(history: list[dict], target: int = 200) -> list[dict]:
    """Downsample a history to about `target` frames.

    Keeps the first 60 entries in full and spreads the remaining slots evenly
    over the rest, always including the final entry.
    """
    if len(history) <= target:
        return history
    dense_upto = min(60, len(history))
    keep = set(range(dense_upto))
    remaining = target - dense_upto
    if remaining > 0:
        keep.update(int(round(t)) for t in
                    np.linspace(dense_upto, len(history) - 1, remaining))
    keep.add(len(history) - 1)
    return [history[i] for i in sorted(keep)]


def write_json(payload: dict, path: str) -> None:
    """Write `payload` to `path` as indented JSON, creating parent directories."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
