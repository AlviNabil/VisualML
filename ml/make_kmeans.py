"""
Fit k-means on the customer dataset and export the results for the SwiftUI app.

Requires Python 3.10+.

Clustering is unsupervised: there is no target to predict, only two features
per customer (annual spend, visit frequency) and the question of whether they
fall into natural groups. `segment` in the CSV is the label the data was
generated from -- it is read here only to check how well the discovered
clusters line up with it, and is never given to k-means.

The export carries:

    dataset          the customers, plus the (unused-by-the-model) true segment
    elbow            final inertia for k = 1..8, best of several restarts each
    restarts         five k-means++ runs at k=4, each with its full history --
                     most converge to the same optimum, one gets stuck in a
                     worse one, which is the point
    best             the lowest-inertia restart: the final centroids ("weights"),
                     assignments, and how they line up with the true segments

Run:  python3.10 ml/make_kmeans.py
"""

import os

import numpy as np

from common import load_dataset, write_json
from kmeans import assign_clusters, descale, fit_kmeans, inertia, standardize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SOURCE_CSV = os.path.join(HERE, "data", "customers.csv")
OUTPUT_JSON = os.path.join(REPO, "Models", "kmeans.json")

K_FOR_ELBOW = range(1, 9)
RESTARTS_PER_ELBOW_K = 8
HEADLINE_K = 4
RESTART_SEEDS = [0, 1, 2, 3, 4]     # exported individually so the app can replay each


def best_of_restarts(X: np.ndarray, k: int, seeds: range | list[int]) -> list[dict]:
    """Run k-means once per seed and return the history of the lowest-inertia run."""
    best_history = None
    for seed in seeds:
        history = fit_kmeans(X, k, seed=seed)
        if best_history is None or history[-1]["inertia"] < best_history[-1]["inertia"]:
            best_history = history
    return best_history


def verify_against_sklearn(Xs: np.ndarray, k: int, our_inertia: float) -> None:
    """Compare our best inertia with scikit-learn's KMeans as a cross-check.

    Different k-means++ implementations do not produce identical centroids, so
    this checks that both reach the same optimum inertia rather than comparing
    centroids directly. Used as a test oracle only; no exported value comes
    from sklearn.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return
    ref = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    gap = abs(our_inertia - ref.inertia_)
    print(f"  cross-check vs sklearn KMeans: "
          f"{'OK' if gap < 1e-2 else 'MISMATCH'} (our={our_inertia:.4f}, "
          f"sklearn={ref.inertia_:.4f})")


def cluster_purity(assignments: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Fraction of points whose cluster's majority true label matches their own."""
    correct = 0
    for c in range(k):
        labels_in_cluster = true_labels[assignments == c]
        if len(labels_in_cluster) == 0:
            continue
        _, counts = np.unique(labels_in_cluster, return_counts=True)
        correct += counts.max()
    return correct / len(true_labels)


def label_clusters(assignments: np.ndarray, true_labels: np.ndarray, k: int) -> list[dict]:
    """For each cluster, the true segment its members most often belong to."""
    labels = []
    for c in range(k):
        members = true_labels[assignments == c]
        if len(members) == 0:
            labels.append({"cluster": c, "majoritySegment": None, "matchRate": 0.0, "size": 0})
            continue
        values, counts = np.unique(members, return_counts=True)
        top = counts.argmax()
        labels.append({
            "cluster": c,
            "majoritySegment": str(values[top]),
            "matchRate": round(float(counts[top] / len(members)), 4),
            "size": int(len(members)),
        })
    return labels


def serialize_history(history: list[dict], mean: np.ndarray, std: np.ndarray) -> list[dict]:
    """Convert one run's history to JSON-safe types, with centroids in real units."""
    return [{
        "iter": step["iter"],
        "centroids": [[round(v, 4) for v in c]
                      for c in descale(step["centroids"], mean, std)],
        "assignments": [int(a) for a in step["assignments"]],
        "inertia": round(step["inertia"], 6),
    } for step in history]


def main() -> None:
    df = load_dataset(SOURCE_CSV)
    X = df[["annual_spend", "visit_frequency"]].to_numpy()
    true_segments = df["segment"].to_numpy()
    print(f"Loaded {len(df)} customers from {os.path.relpath(SOURCE_CSV, REPO)}")

    Xs, mean, std = standardize(X)
    print(f"Standardized: mean={np.round(mean, 2)}, std={np.round(std, 2)}")

    # ---- Elbow curve: best-of-restarts inertia for each k -------------------
    print(f"\nElbow curve ({RESTARTS_PER_ELBOW_K} restarts per k):")
    elbow = []
    for k in K_FOR_ELBOW:
        history = best_of_restarts(Xs, k, range(RESTARTS_PER_ELBOW_K))
        final = history[-1]
        elbow.append({
            "k": k,
            "inertia": round(final["inertia"], 6),
            "centroids": [[round(v, 4) for v in c] for c in descale(final["centroids"], mean, std)],
            "assignments": [int(a) for a in final["assignments"]],
        })
        print(f"  k={k}  inertia={final['inertia']:8.4f}")

    # ---- Individual restarts at the headline k, kept separately -------------
    print(f"\nFive individual restarts at k={HEADLINE_K} (no best-of selection):")
    restarts = []
    for seed in RESTART_SEEDS:
        history = fit_kmeans(Xs, HEADLINE_K, seed=seed)
        final = history[-1]
        purity = cluster_purity(np.array(final["assignments"]), true_segments, HEADLINE_K)
        print(f"  seed={seed}  iterations={final['iter']:3d}  "
              f"inertia={final['inertia']:8.4f}  purity={purity:.3f}")
        restarts.append({
            "seed": seed,
            "iterations": final["iter"],
            "finalInertia": round(final["inertia"], 6),
            "purity": round(purity, 4),
            "history": serialize_history(history, mean, std),
        })

    best_run = min(restarts, key=lambda r: r["finalInertia"])
    best_final = best_run["history"][-1]
    best_assignments = np.array(best_final["assignments"])
    print(f"\nBest restart: seed={best_run['seed']}, "
          f"inertia={best_run['finalInertia']:.4f}, purity={best_run['purity']:.3f}")
    verify_against_sklearn(Xs, HEADLINE_K, best_run["finalInertia"])

    cluster_labels = label_clusters(best_assignments, true_segments, HEADLINE_K)
    print("Cluster identities (majority vote against the true segment, unused by the fit):")
    for entry in cluster_labels:
        print(f"  cluster {entry['cluster']}: {entry['size']:3d} customers, "
              f"mostly {entry['majoritySegment']} ({entry['matchRate']:.0%})")

    payload = {
        "model": "kmeans",
        "title": "K-Means Clustering",
        "subtitle": "Grouping customers by spend and visit frequency",
        "library": "hand-written numpy (pandas for I/O, sklearn as a cross-check oracle only)",
        "k": HEADLINE_K,
        "dataset": {
            "name": "customers",
            "featureNames": ["annual_spend", "visit_frequency"],
            "axisLabels": ["Annual spend ($k)", "Visit frequency (per month)"],
            "n": len(df),
            "mins": [float(v) for v in X.min(axis=0)],
            "maxs": [float(v) for v in X.max(axis=0)],
            "points": [{"x": [float(a), float(b)], "trueSegment": str(seg)}
                       for a, b, seg in zip(X[:, 0], X[:, 1], true_segments)],
        },
        "elbow": elbow,
        "restarts": restarts,
        "best": {
            "seed": best_run["seed"],
            "centroids": best_final["centroids"],
            "assignments": best_final["assignments"],
            "inertia": best_run["finalInertia"],
            "purity": best_run["purity"],
            "clusterLabels": cluster_labels,
        },
    }

    print()
    write_json(payload, OUTPUT_JSON)


if __name__ == "__main__":
    main()
