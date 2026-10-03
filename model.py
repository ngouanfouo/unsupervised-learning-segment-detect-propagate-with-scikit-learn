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

# Step 13 - propagate_and_train (not yet solved)
# TODO: implement

# Step 14 - synthetic_image (not yet solved)
# TODO: implement

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

