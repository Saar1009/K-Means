import sys, os
import symnmf as sm 
from sklearn.metrics import silhouette_score as skl_silhouette_score
import math

def handleError():
    print("An Error Has Occurred")
    sys.exit(1)


# ------------------- args & IO ----------------------

def parse_args():
    """parse CLI args (k, filename) """
    if len(sys.argv) != 3:
        print("Invalid Input!")
        sys.exit(1)
    a1, a2 = sys.argv[1], sys.argv[2]
    if a1.isdigit() and int(a1) > 0:
        return int(a1), a2
    if a2.isdigit() and int(a2) > 0:
        return int(a2), a1
    print("Invalid Input!")
    sys.exit(1)


def load_points(file_name):
    """load a CSV of points into List[List[float]] and validates dimension """
    pts = []
    try:
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [float(tok) for tok in line.split(",")]
                pts.append(row)
    except Exception:
        raise
    if not pts or not pts[0]:
        raise ValueError("empty input")
    d = len(pts[0])
    for r in pts:
        if len(r) != d:
            raise ValueError("mismatch row length")
    return pts


def create_d_vectors(file_obj):
    """converts a CSV file into python list[list[float]]"""
    X = []
    for line in file_obj:
        line = line.strip()
        if not line:
            continue
        X.append([float(tok) for tok in line.split(",")])
    return X


# ------------------- symnmf path ----------------------

def run_symnmf(file_name, k):
    """compute H via symnmf: builds N and H0 in python and delegates to c core """
    return sm.symnmf_from_file(file_name, k)


def argmax_labels(H):
    """convert soft assignment matrix H (n x k) to hard labels via argmax per row """
    labs = []
    for row in H:
        if not row:
            labs.append(0)
            continue
        jmax, vmax = 0, row[0]
        for j in range(1, len(row)):
            v = row[j]
            if v > vmax:
                vmax, jmax = v, j
        labs.append(jmax)
    return labs


def symnmf_clusters(K, X, filename):
    """compute hard labels via symnmf using the module API that reads from file"""
    H = sm.symnmf_from_file(filename, K)
    return argmax_labels(H)


# ------------------- K-Means (HW1-style) ----------------------

def euclidean(p1, p2):
    """euclidean distance between two vectors """
    s = 0.0
    for a, b in zip(p1, p2):
        d = a - b
        s += d * d
    return s ** 0.5


def assign_clusters(data, centroids):
    """assign each point to its nearest centroid """
    clusters = [[] for _ in centroids]
    labels = [0] * len(data)
    for i, x in enumerate(data):
        best, bid = euclidean(x, centroids[0]), 0
        for j in range(1, len(centroids)):
            d = euclidean(x, centroids[j])
            if d < best:
                best, bid = d, j
        clusters[bid].append(x)
        labels[i] = bid
    return clusters, labels


def update_centroids(clusters, dim):
    """recompute centroid of each cluster as the mean of its points """
    new = []
    for c in clusters:
        if not c:
            new.append([0.0] * dim)
            continue
        z = [0.0] * dim
        inv = 1.0 / float(len(c))
        for p in c:
            for i in range(dim):
                z[i] += p[i] * inv
        new.append(z)
    return new


def has_converged(old, new, eps):
    """return true if all centroid moves are smaller than eps """
    for a, b in zip(old, new):
        if euclidean(a, b) >= eps:
            return False
    return True


def kmeans(points, k, max_iters=300, eps=1e-4):
    """minimal K - Means (init = first k points) """
    centroids = [row[:] for row in points[:k]]
    dim = len(points[0])
    for _ in range(max_iters):
        clusters, labels = assign_clusters(points, centroids)
        new_c = update_centroids(clusters, dim)
        if has_converged(centroids, new_c, eps):
            centroids = new_c
            break
        centroids = new_c
    _, labels = assign_clusters(points, centroids)
    return labels


def kmeans_clusters(K, X):
    return kmeans(X, K)


# ------------------- glue ----------------------

def print_scores(s_symnmf, s_kmeans):
    """print two scores with 4 decimals """
    print("nmf: {:.4f}".format(float(s_symnmf)))
    print("kmeans: {:.4f}".format(float(s_kmeans)))


def main():
    """performs analysis and compare between kmeans and symnmf"""
    if len(sys.argv) != 3:
        handleError()
    try:
        K = int(sys.argv[1])
        if K <= 1:
            handleError()
        filename = sys.argv[2]
    except Exception:
        handleError()
    try:
        with open(filename, 'r') as file:
            X = create_d_vectors(file)
    except Exception:
        handleError()

    kmeans_cluster = kmeans_clusters(K, X)
    symnmf_cluster = symnmf_clusters(K, X, filename)

    try:
        nmf_score = float(skl_silhouette_score(X, symnmf_cluster))
    except Exception:     
        nmf_score = 0.0 # crash protection
    try:
        km_score = float(skl_silhouette_score(X, kmeans_cluster))
    except Exception: 
        km_score = 0.0 # crash protection

    print("nmf: %.4f" % nmf_score)
    print("kmeans: %.4f" % km_score)


if __name__ == "__main__":
    main()