import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

class KMeans:
    def __init__(self, n_clusters, max_iters=100, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.history = []

    def fit(self, X):
        self.centroids = X[np.random.choice(range(X.shape[0]), self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(X)
            old_centroids = self.centroids
            self.centroids = self._get_centroids(X)
            self.history.append((self.centroids, self.clusters))

            if self._is_converged(old_centroids, self.centroids):
                break

    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample):
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.n_clusters)]
        return sum(distances) < self.tol

    def predict(self, X):
        cluster_labels = np.zeros(X.shape[0])
        for idx, sample in enumerate(X):
            cluster_idx = self._closest_centroid(sample)
            cluster_labels[idx] = cluster_idx
        return cluster_labels