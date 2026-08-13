"""
Neural network math, used by make_neural.py.

Requires Python 3.10+.

A plain feed-forward network trained by full-batch gradient descent. Every
intermediate quantity is kept rather than discarded, because the whole point of
the stage is to show what normally stays hidden: the pre-activation z and the
activation a at each layer on the way forward, and every gradient on the way
back.

Shapes, for a batch of n rows:
    a[0]  = X            (n, d_in)
    z[l]  = a[l-1] @ W[l] + b[l]      (n, d_l)
    a[l]  = f(z[l])                   (n, d_l)

Division of labour:
  * pandas       -- reading the CSV
  * scikit-learn -- used only as a cross-check oracle on final accuracy
  * numpy + this file -- all of the mathematics
"""

import numpy as np

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
#
# Each entry pairs f(z) with f'(z). The derivative is what backpropagation
# multiplies by when it passes an error signal back through a layer, so a
# function whose derivative is near zero over a wide range will stall training
# there -- which is exactly what the app visualizes.


def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-z), squashing the reals into (0, 1)."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """s(z)*(1 - s(z)), which peaks at 0.25 and vanishes in both tails."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent, squashing the reals into (-1, 1)."""
    return np.tanh(z)


def tanh_prime(z: np.ndarray) -> np.ndarray:
    """1 - tanh(z)^2, which peaks at 1.0 and vanishes in both tails."""
    return 1.0 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    """max(0, z): pass positives through untouched, clamp negatives to zero."""
    return np.maximum(0.0, z)


def relu_prime(z: np.ndarray) -> np.ndarray:
    """1 where z > 0, else 0. Constant on the positive side, so no vanishing."""
    return (z > 0).astype(float)


def leaky_relu(z: np.ndarray, slope: float = 0.1) -> np.ndarray:
    """Like ReLU, but negatives keep a small slope instead of going flat."""
    return np.where(z > 0, z, slope * z)


def leaky_relu_prime(z: np.ndarray, slope: float = 0.1) -> np.ndarray:
    """1 where z > 0, else `slope`, so a negative unit can still learn."""
    return np.where(z > 0, 1.0, slope)


def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Linear for positive z, saturating smoothly to -alpha for negative z."""
    return np.where(z > 0, z, alpha * (np.exp(np.minimum(z, 0)) - 1))


def elu_prime(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """1 for positive z, alpha*e^z below zero."""
    return np.where(z > 0, 1.0, alpha * np.exp(np.minimum(z, 0)))


def gelu(z: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit, in its tanh approximation.

    A smooth version of ReLU that dips slightly below zero near the origin.
    """
    inner = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)
    return 0.5 * z * (1.0 + np.tanh(inner))


def gelu_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximated GELU."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (z + 0.044715 * z**3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t**2) * c * (1.0 + 3 * 0.044715 * z**2)


ACTIVATIONS: dict[str, dict] = {
    "sigmoid": {
        "fn": sigmoid, "prime": sigmoid_prime,
        "formula": "f(z) = 1 / (1 + e^-z)",
        "derivative": "f'(z) = f(z)·(1 - f(z))",
        "range": "(0, 1)",
        "note": "Saturates at both ends, where the derivative falls to almost "
                "zero and learning stalls. Still the natural choice for a "
                "binary output, where a probability is wanted.",
    },
    "tanh": {
        "fn": tanh, "prime": tanh_prime,
        "formula": "f(z) = tanh(z)",
        "derivative": "f'(z) = 1 - tanh(z)²",
        "range": "(-1, 1)",
        "note": "Zero-centred, and its peak slope is 1.0 against sigmoid's "
                "0.25, so gradients survive better through a stack of layers. "
                "Still saturates in both tails.",
    },
    "relu": {
        "fn": relu, "prime": relu_prime,
        "formula": "f(z) = max(0, z)",
        "derivative": "f'(z) = 1 if z > 0 else 0",
        "range": "[0, ∞)",
        "note": "No saturation on the positive side, so deep stacks train far "
                "faster. A unit pushed permanently negative has zero gradient "
                "for every input and stops learning -- a 'dead' unit.",
    },
    "leaky_relu": {
        "fn": leaky_relu, "prime": leaky_relu_prime,
        "formula": "f(z) = z if z > 0 else 0.1·z",
        "derivative": "f'(z) = 1 if z > 0 else 0.1",
        "range": "(-∞, ∞)",
        "note": "ReLU with a shallow slope kept on the negative side, so a "
                "unit that goes negative still receives some gradient and can "
                "recover.",
    },
    "elu": {
        "fn": elu, "prime": elu_prime,
        "formula": "f(z) = z if z > 0 else e^z - 1",
        "derivative": "f'(z) = 1 if z > 0 else e^z",
        "range": "(-1, ∞)",
        "note": "Smooth at the origin and saturates gently to -1, which keeps "
                "activations closer to zero-centred than ReLU does.",
    },
    "gelu": {
        "fn": gelu, "prime": gelu_prime,
        "formula": "f(z) ≈ 0.5·z·(1 + tanh(√(2/π)(z + 0.044715 z³)))",
        "derivative": "f'(z) — smooth, dips below 0 near the origin",
        "range": "(-0.17, ∞)",
        "note": "A smooth ReLU that weights the input by how likely it is to "
                "be kept. The default in most current transformer models.",
    },
}


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


def initialize(layer_sizes: list[int], seed: int) -> list[dict]:
    """Create weight matrices and bias vectors for each layer.

    Uses Xavier scaling -- weights drawn with standard deviation
    sqrt(1 / fan_in) -- so the signal entering each layer starts at roughly
    unit scale instead of exploding or dying out with depth.
    """
    rng = np.random.default_rng(seed)
    layers = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append({
            "W": rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, fan_out)),
            "b": np.zeros(fan_out),
        })
    return layers


def forward(X: np.ndarray, layers: list[dict], hidden: str) -> dict:
    """Run the network forward, keeping every intermediate value.

    The hidden layers use `hidden`; the output layer always uses sigmoid, so
    the result reads as a probability.

    Returns a dict with the per-layer pre-activations z and activations a.
    """
    activation = ACTIVATIONS[hidden]["fn"]
    zs, activations = [], [X]
    a = X

    for index, layer in enumerate(layers):
        z = a @ layer["W"] + layer["b"]
        last = index == len(layers) - 1
        a = sigmoid(z) if last else activation(z)
        zs.append(z)
        activations.append(a)

    return {"zs": zs, "activations": activations, "output": activations[-1]}


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    """-(1/n) Σ [ y·log(p) + (1-y)·log(1-p) ]

    Both arguments are flattened first: predictions arrive as a column (n, 1)
    and labels as a row (n,), and leaving that mismatched would broadcast them
    into an (n, n) matrix and average over the wrong thing entirely.
    """
    eps = 1e-12
    y = np.asarray(y).ravel()
    p = np.clip(np.asarray(p).ravel(), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def backward(y: np.ndarray, layers: list[dict], pass_: dict, hidden: str) -> dict:
    """Propagate the error backward, keeping every gradient.

    Starting at the output, where pairing sigmoid with cross-entropy collapses
    the two derivatives into a single term:

        dL/dz_last = (p - y) / n

    then repeatedly, moving one layer back:

        dL/dW[l] = a[l-1]ᵀ · dL/dz[l]
        dL/db[l] = Σ dL/dz[l]
        dL/da[l-1] = dL/dz[l] · W[l]ᵀ
        dL/dz[l-1] = dL/da[l-1] ⊙ f'(z[l-1])

    That last multiplication by f'(z) is where the choice of activation decides
    whether the signal survives the trip backward.

    Returns per-layer dicts of dW, db, dz and da, ordered front to back.
    """
    prime = ACTIVATIONS[hidden]["prime"]
    n = y.shape[0]
    zs, activations = pass_["zs"], pass_["activations"]

    grads = [None] * len(layers)
    # The output layer's shortcut: sigmoid' cancels against the log in the loss.
    dz = (pass_["output"] - y.reshape(-1, 1)) / n

    for index in reversed(range(len(layers))):
        a_prev = activations[index]
        grads[index] = {
            "dW": a_prev.T @ dz,
            "db": dz.sum(axis=0),
            "dz": dz,
        }
        if index > 0:
            da_prev = dz @ layers[index]["W"].T
            grads[index]["da_prev"] = da_prev
            dz = da_prev * prime(zs[index - 1])

    return grads


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Share of rows whose predicted class matches the label at a 0.5 cut."""
    return float(np.mean((p.ravel() >= 0.5).astype(int) == y))


def train(X: np.ndarray, y: np.ndarray, layer_sizes: list[int], hidden: str,
          learning_rate: float, epochs: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Train by full-batch gradient descent, recording the model each epoch.

    Returns (layers, history) where layers is the trained network and history
    holds one entry per epoch with the loss, the accuracy, and a copy of the
    weights at that point.
    """
    layers = initialize(layer_sizes, seed)
    history = []

    for epoch in range(epochs + 1):
        pass_ = forward(X, layers, hidden)
        loss = binary_cross_entropy(y, pass_["output"])

        history.append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy(y, pass_["output"]),
            "layers": [{"W": layer["W"].copy(), "b": layer["b"].copy()}
                       for layer in layers],
        })

        if epoch == epochs:
            break

        grads = backward(y, layers, pass_, hidden)
        for layer, grad in zip(layers, grads):
            layer["W"] -= learning_rate * grad["dW"]
            layer["b"] -= learning_rate * grad["db"]

    return layers, history
