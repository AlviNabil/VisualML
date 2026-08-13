"""
Train a small neural network on the ring dataset and export every intermediate
value for the SwiftUI app.

Requires Python 3.10+.

The network is deliberately tiny -- 2 inputs, two hidden layers of 4, one
output -- so that every weight, every pre-activation and every gradient can be
put on screen as an actual number rather than summarised away.

The export carries one section per stage of the walkthrough:

    dataset         the rings, and a logistic-regression baseline that fails
    architecture    layer sizes, weight-matrix shapes, parameter count
    forwardTrace    one point carried through the network, layer by layer
    backwardTrace   the gradients for that same point, flowing back
    activations     f and f' sampled for six activation functions, plus a
                    training run per activation so their speeds compare
    training        loss, accuracy, weights and decision-boundary grids
    neurons         what each hidden unit responds to, once trained

Run:  python3.10 ml/make_neural.py
"""

import os

import numpy as np

from common import load_dataset, write_json
from neural import (ACTIVATIONS, accuracy, backward, binary_cross_entropy,
                    forward, train)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SOURCE_CSV = os.path.join(HERE, "data", "rings.csv")
OUTPUT_JSON = os.path.join(REPO, "Models", "neural.json")

LAYER_SIZES = [2, 4, 4, 1]
MAIN_ACTIVATION = "tanh"
LEARNING_RATE = 0.5
EPOCHS = 3000
GRID = 34               # resolution of the decision-boundary map
BOUNDARY_FRAMES = 24    # how many epochs to keep a full map for
CURVE_POINTS = 121      # samples per activation function plot


def round_list(values, places: int = 4):
    """Recursively round nested numeric arrays for a compact export."""
    array = np.asarray(values, dtype=float)
    return np.round(array, places).tolist()


def checkpoint_epochs(total: int, count: int) -> list[int]:
    """Pick epochs to keep, dense early where the model changes fastest."""
    dense = list(range(0, min(12, total + 1)))
    rest = np.unique(np.round(np.geomspace(12, total, count - len(dense))).astype(int))
    return sorted(set(dense) | set(int(e) for e in rest))


def decision_grid(layers, hidden: str, lo: float, hi: float, steps: int) -> list[list[float]]:
    """Evaluate the network over a square grid, giving the probability map."""
    axis = np.linspace(lo, hi, steps)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    probs = forward(points, layers, hidden)["output"].reshape(steps, steps)
    return round_list(probs, 3)


def build_forward_trace(x: np.ndarray, label: int, layers, hidden: str) -> dict:
    """Carry a single point through the network, keeping each step."""
    row = x.reshape(1, -1)
    pass_ = forward(row, layers, hidden)

    steps = []
    for index, layer in enumerate(layers):
        last = index == len(layers) - 1
        steps.append({
            "layer": index + 1,
            "activation": "sigmoid" if last else hidden,
            "input": round_list(pass_["activations"][index].ravel()),
            "weights": round_list(layer["W"]),
            "biases": round_list(layer["b"]),
            "z": round_list(pass_["zs"][index].ravel()),
            "a": round_list(pass_["activations"][index + 1].ravel()),
        })

    prediction = float(pass_["output"].ravel()[0])
    return {
        "input": round_list(x),
        "label": int(label),
        "steps": steps,
        "prediction": round(prediction, 4),
        "loss": round(binary_cross_entropy(np.array([label]), pass_["output"]), 4),
    }


def build_backward_trace(x: np.ndarray, label: int, layers, hidden: str) -> dict:
    """Propagate that same point's error back, keeping each gradient."""
    row = x.reshape(1, -1)
    labels = np.array([label])
    pass_ = forward(row, layers, hidden)
    grads = backward(labels, layers, pass_, hidden)

    steps = []
    for index in reversed(range(len(layers))):
        entry = {
            "layer": index + 1,
            "dz": round_list(grads[index]["dz"].ravel()),
            "dW": round_list(grads[index]["dW"]),
            "db": round_list(grads[index]["db"]),
        }
        if "da_prev" in grads[index]:
            entry["daPrev"] = round_list(grads[index]["da_prev"].ravel())
            entry["primeZ"] = round_list(
                ACTIVATIONS[hidden]["prime"](pass_["zs"][index - 1]).ravel())
        steps.append(entry)

    return {
        "prediction": round(float(pass_["output"].ravel()[0]), 4),
        "label": int(label),
        "outputError": round(float(pass_["output"].ravel()[0] - label), 4),
        "steps": steps,
    }


def build_activation_samples() -> list[dict]:
    """Sample f and f' for every activation, plus how fast each one trains."""
    xs = np.linspace(-5, 5, CURVE_POINTS)
    samples = []
    for name, spec in ACTIVATIONS.items():
        samples.append({
            "name": name,
            "formula": spec["formula"],
            "derivative": spec["derivative"],
            "range": spec["range"],
            "note": spec["note"],
            "x": round_list(xs, 3),
            "y": round_list(spec["fn"](xs), 4),
            "dy": round_list(spec["prime"](xs), 4),
            "maxSlope": round(float(np.max(spec["prime"](xs))), 4),
        })
    return samples


def main() -> None:
    df = load_dataset(SOURCE_CSV)
    X = df[["x1", "x2"]].to_numpy()
    y = df["label"].to_numpy()
    print(f"Loaded {len(df)} points from {os.path.relpath(SOURCE_CSV, REPO)}")

    # ---- A straight line, for contrast ------------------------------------
    from sklearn.linear_model import LogisticRegression
    line = LogisticRegression().fit(X, y)
    line_accuracy = float(line.score(X, y))
    print(f"\nLogistic regression (one straight boundary): {line_accuracy:.4f}")

    # ---- The main run ------------------------------------------------------
    print(f"\nTraining {LAYER_SIZES} with {MAIN_ACTIVATION}, "
          f"lr={LEARNING_RATE}, {EPOCHS} epochs")
    layers, history = train(X, y, LAYER_SIZES, MAIN_ACTIVATION,
                            LEARNING_RATE, EPOCHS, seed=1)
    final = history[-1]
    print(f"  final loss {final['loss']:.4f}   accuracy {final['accuracy']:.4f}")

    try:
        from sklearn.neural_network import MLPClassifier
        ref = MLPClassifier(tuple(LAYER_SIZES[1:-1]), activation="tanh",
                            max_iter=8000, random_state=0).fit(X, y).score(X, y)
        print(f"  cross-check vs sklearn MLPClassifier: {ref:.4f}")
    except ImportError:
        pass

    lo, hi = float(X.min()) - 0.4, float(X.max()) + 0.4
    keep = checkpoint_epochs(EPOCHS, BOUNDARY_FRAMES)
    frames = []
    for epoch in keep:
        entry = history[epoch]
        frames.append({
            "epoch": epoch,
            "loss": round(entry["loss"], 6),
            "accuracy": round(entry["accuracy"], 4),
            "weights": [round_list(layer["W"]) for layer in entry["layers"]],
            "biases": [round_list(layer["b"]) for layer in entry["layers"]],
            "grid": decision_grid(entry["layers"], MAIN_ACTIVATION, lo, hi, GRID),
        })
    print(f"  kept {len(frames)} boundary frames on a {GRID}x{GRID} grid")

    # The full loss curve, thinned so the chart stays smooth but small.
    curve_epochs = sorted(set(list(range(0, 60)) + list(range(60, EPOCHS + 1, 20))))
    loss_curve = [{"epoch": e,
                   "loss": round(history[e]["loss"], 6),
                   "accuracy": round(history[e]["accuracy"], 4)}
                  for e in curve_epochs]

    # ---- Forward and backward traces, on one representative point ----------
    # Pick a point sitting well inside the outer ring, so the numbers are not
    # borderline and each step reads clearly.
    radius = np.sqrt((X ** 2).sum(axis=1))
    candidates = np.where(y == 1)[0]
    chosen = int(candidates[np.argsort(np.abs(radius[candidates] - np.median(radius[candidates])))[0]])
    print(f"\nTracing point #{chosen}: x={np.round(X[chosen], 3)}, label={y[chosen]}")

    trained_layers = [{"W": np.array(layer["W"]), "b": np.array(layer["b"])}
                      for layer in history[-1]["layers"]]
    forward_trace = build_forward_trace(X[chosen], int(y[chosen]),
                                        trained_layers, MAIN_ACTIVATION)
    print(f"  forward  -> prediction {forward_trace['prediction']:.4f}")

    # Backprop is shown on a partly-trained network, where the error is still
    # large enough for the gradients to be worth looking at.
    mid_epoch = next(e for e in keep if e >= 12)
    mid_layers = [{"W": np.array(layer["W"]), "b": np.array(layer["b"])}
                  for layer in history[mid_epoch]["layers"]]
    backward_trace = build_backward_trace(X[chosen], int(y[chosen]),
                                          mid_layers, MAIN_ACTIVATION)
    backward_trace["epoch"] = mid_epoch
    forward_at_mid = build_forward_trace(X[chosen], int(y[chosen]),
                                         mid_layers, MAIN_ACTIVATION)
    print(f"  backward -> at epoch {mid_epoch}, prediction "
          f"{backward_trace['prediction']:.4f}, error {backward_trace['outputError']:+.4f}")

    # ---- Every activation, sampled and raced --------------------------------
    print("\nActivation comparison (same architecture, same seed):")
    activation_samples = build_activation_samples()
    for sample in activation_samples:
        name = sample["name"]
        _, hist = train(X, y, LAYER_SIZES, name, LEARNING_RATE, EPOCHS, seed=1)
        reached = next((h["epoch"] for h in hist if h["accuracy"] >= 0.95), None)
        sample["training"] = {
            "finalLoss": round(hist[-1]["loss"], 6),
            "finalAccuracy": round(hist[-1]["accuracy"], 4),
            "epochsTo95": reached,
            "curve": [{"epoch": e, "loss": round(hist[e]["loss"], 6)}
                      for e in curve_epochs],
        }
        print(f"  {name:11s} loss {hist[-1]['loss']:.4f}  "
              f"acc {hist[-1]['accuracy']:.4f}  95% at epoch {reached}")

    # ---- What each hidden unit ended up responding to -----------------------
    axis = np.linspace(lo, hi, GRID)
    xx, yy = np.meshgrid(axis, axis)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    trained_pass = forward(grid_points, trained_layers, MAIN_ACTIVATION)
    neurons = []
    for layer_index in (0, 1):
        activations = trained_pass["activations"][layer_index + 1]
        for unit in range(activations.shape[1]):
            neurons.append({
                "layer": layer_index + 1,
                "unit": unit,
                "grid": round_list(activations[:, unit].reshape(GRID, GRID), 3),
            })
    print(f"\nCaptured {len(neurons)} hidden-unit response maps")

    payload = {
        "model": "neural",
        "title": "Neural Network",
        "subtitle": "Forward pass, backpropagation, and what activations do",
        "library": "hand-written numpy (pandas for I/O, sklearn as a cross-check oracle only)",
        "dataset": {
            "name": "rings",
            "featureNames": ["x1", "x2"],
            "axisLabels": ["Feature 1", "Feature 2"],
            "classNames": ["Inner", "Outer"],
            "n": len(df),
            "min": lo, "max": hi,
            "points": [{"x": [float(a), float(b)], "label": int(c)}
                       for a, b, c in zip(X[:, 0], X[:, 1], y)],
            "linearBaselineAccuracy": round(line_accuracy, 4),
        },
        "architecture": {
            "layerSizes": LAYER_SIZES,
            "hiddenActivation": MAIN_ACTIVATION,
            "outputActivation": "sigmoid",
            "learningRate": LEARNING_RATE,
            "epochs": EPOCHS,
            "parameterCount": int(sum(a * b + b for a, b in
                                      zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]))),
        },
        "forwardTrace": forward_trace,
        "forwardTraceAtBackprop": forward_at_mid,
        "backwardTrace": backward_trace,
        "activations": activation_samples,
        "training": {
            "gridSize": GRID,
            "gridMin": lo, "gridMax": hi,
            "curve": loss_curve,
            "frames": frames,
            "finalLoss": round(final["loss"], 6),
            "finalAccuracy": round(final["accuracy"], 4),
        },
        "neurons": neurons,
    }

    print()
    write_json(payload, OUTPUT_JSON)


if __name__ == "__main__":
    main()
