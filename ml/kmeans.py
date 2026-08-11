"""
K-means clustering math, used by make_kmeans.py.

Requires Python 3.10+.

Division of labour:
  * pandas       -- reading the CSV (via common.load_dataset)
  * scikit-learn -- used only as a cross-check oracle on the final inertia
  * numpy + this file -- all of the clustering mathematics
"""

import numpy as np


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rescale each column of X to zero mean and unit variance.

    Returns (X_scaled, mean, std). A zero std is replaced with 1 so the
    division never produces NaN.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (X - mean) / std, mean, std


def descale(points: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert `standardize`, mapping standardized points back to original units."""
    return points * std + mean


def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose k initial centroids from the rows of X using k-means++.

    The first centroid is drawn uniformly at random. Each following centroid
    is drawn from the remaining points with probability proportional to the
    squared distance to the nearest centroid already chosen.
    """
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(n)]
    closest_sq = ((X - centroids[0]) ** 2).sum(axis=1)

    for i in range(1, k):
        probs = closest_sq / closest_sq.sum()
        centroids[i] = X[rng.choice(n, p=probs)]
        new_sq = ((X - centroids[i]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    return centroids


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row of X to the index of its nearest centroid."""
    dist_sq = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist_sq, axis=1)


def update_centroids(X: np.ndarray, assignments: np.ndarray, k: int,
                     previous: np.ndarray) -> np.ndarray:
    """Recompute each centroid as the mean of the points assigned to it.

    A cluster with no points assigned keeps its previous position.
    """
    centroids = previous.copy()
    for c in range(k):
        members = X[assignments == c]
        if len(members) > 0:
            centroids[c] = members.mean(axis=0)
    return centroids


def inertia(X: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
    """Total within-cluster sum of squared distances: sum( ||x - centroid||^2 )."""
    return float(((X - centroids[assignments]) ** 2).sum())


def fit_kmeans(X: np.ndarray, k: int, seed: int,
              max_iters: int = 100, tol: float = 1e-6) -> list[dict]:
    """Run k-means to convergence from a k-means++ start, recording every iteration.

    Loop:
        assign each point to its nearest centroid
        move each centroid to the mean of its assigned points
    Stops when the largest centroid movement is below `tol`, or after
    `max_iters` steps.

    Returns a list of dicts, one per iteration, each holding the centroids,
    the assignments and the inertia at that point.
    """
    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus_init(X, k, rng)
    assignments = assign_clusters(X, centroids)

    history = [{
        "iter": 0,
        "centroids": centroids.copy(),
        "assignments": assignments.copy(),
        "inertia": inertia(X, assignments, centroids),
    }]

    for i in range(1, max_iters + 1):
        new_centroids = update_centroids(X, assignments, k, centroids)
        shift = float(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max())
        centroids = new_centroids
        assignments = assign_clusters(X, centroids)

        history.append({
            "iter": i,
            "centroids": centroids.copy(),
            "assignments": assignments.copy(),
            "inertia": inertia(X, assignments, centroids),
        })

        if shift < tol:
            break

    return history
