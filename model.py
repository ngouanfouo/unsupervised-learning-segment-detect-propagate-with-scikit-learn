"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - blobs_data
from sklearn.datasets import make_blobs

def blobs_data(random_state=42):
    centers = [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y

# Step 2 - fit_kmeans
from sklearn.cluster import KMeans

def fit_kmeans(X, k, random_state=42):
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)

# Step 3 - inertia_curve
def inertia_curve(X, ks):
    return {k: float(fit_kmeans(X, k).inertia_) for k in ks}

# Step 4 - silhouette_curve
from sklearn.metrics import silhouette_score


def silhouette_curve(X, ks):
    return {
        k: float(silhouette_score(X, fit_kmeans(X, k).labels_))
        for k in ks
        if k >= 2
    }


def best_k_by_silhouette(curve):
    return max(curve, key=lambda k: (curve[k], -k))

# Step 5 - fit_dbscan
from sklearn.cluster import DBSCAN


def fit_dbscan(X, eps=0.2, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    return {
        "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "n_noise": int((labels == -1).sum()),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }

# Step 6 - dbscan_predict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_y = dbscan.labels_[dbscan.core_sample_indices_]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(core_X, core_y)
    return knn.predict(X_new).astype(int)

# Step 7 - fit_gmm
from sklearn.mixture import GaussianMixture


def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    ).fit(X)


def bic_curve(X, ks):
    return {k: float(fit_gmm(X, k).bic(X)) for k in ks}

# Step 8 - flag_anomalies
import numpy as np


def flag_anomalies(gmm, X, contamination=0.04):
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)
    return densities < threshold

# Step 9 - digits_data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def digits_data(test_size=0.25, random_state=42):
    X, y = load_digits(return_X_y=True)
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

# Step 10 - baseline_50_random
from sklearn.linear_model import LogisticRegression


def baseline_50_random(X_train, y_train, X_test, y_test, n_labeled=50, random_state=42):
    X_lab = X_train[:n_labeled]
    y_lab = y_train[:n_labeled]
    clf = LogisticRegression(max_iter=10000).fit(X_lab, y_lab)
    return float(clf.score(X_test, y_test))

# Step 11 - representative_digits
import numpy as np


def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)
    distances = kmeans.transform(X_train)
    rep_idx = np.argmin(distances, axis=0)
    return kmeans, rep_idx

# Step 12 - train_on_representatives
from sklearn.linear_model import LogisticRegression


def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    clf = LogisticRegression(max_iter=10000).fit(
        X_train[rep_idx], y_train[rep_idx]
    )
    return float(clf.score(X_test, y_test))

# Step 13 - propagate_and_train
import numpy as np
from sklearn.linear_model import LogisticRegression


def propagate_and_train(X_train, y_train, kmeans, rep_idx, X_test, y_test, percentile=20):
    distances = kmeans.transform(X_train)          # (n, k)
    labels = kmeans.labels_                        # cluster of each point
    k = len(rep_idx)

    # Distance from each point to its own centroid
    d_own = distances[np.arange(len(X_train)), labels]

    selected_idx = []
    propagated_y = []

    for j in range(k):
        members = np.where(labels == j)[0]
        if len(members) == 0:
            continue
        cutoff = np.percentile(d_own[members], percentile)
        chosen = members[d_own[members] <= cutoff]
        selected_idx.append(chosen)
        propagated_y.append(np.full(len(chosen), y_train[rep_idx[j]]))

    selected_idx = np.concatenate(selected_idx)
    propagated_y = np.concatenate(propagated_y).astype(y_train.dtype)

    # How many of the propagated labels are actually correct?
    label_accuracy = float((propagated_y == y_train[selected_idx]).mean())

    clf = LogisticRegression(max_iter=10000).fit(
        X_train[selected_idx], propagated_y
    )
    test_accuracy = float(clf.score(X_test, y_test))

    return {
        "n_propagated": int(len(selected_idx)),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }

# Step 14 - synthetic_image
import numpy as np


def synthetic_image(size=48):
    img = np.zeros((size, size, 3), dtype=np.float64)
    half = size // 2

    # Green channel rises 0 -> 1 across each half
    green = np.linspace(0.0, 1.0, half)

    # Left half: red -> yellow  (R = 1, G = green, B = 0)
    img[:, :half, 0] = 1.0
    img[:, :half, 1] = green
    img[:, :half, 2] = 0.0

    # Right half: blue -> cyan   (R = 0, G = green, B = 1)
    img[:, half:, 0] = 0.0
    img[:, half:, 1] = green
    img[:, half:, 2] = 1.0

    # Pure white square in the top-left corner
    sq = size // 4
    img[:sq, :sq, :] = 1.0

    return img

# Step 15 - segment_colors
def segment_colors(image, k=4, random_state=42):
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    centers = kmeans.cluster_centers_[kmeans.labels_]   # (h*w, 3)
    segmented = centers.reshape(h, w, 3)
    return segmented

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

