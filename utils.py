import numpy as np

# K-MEANS MANUAL 
def kmeans_manual(data, k, max_iter=100):
    np.random.seed(42)
    centroids = data.sample(n=k).to_numpy()
    
    for _ in range(max_iter):
        distances = np.linalg.norm(data.to_numpy()[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        old_centroids = centroids.copy()
        centroids = np.array([data.to_numpy()[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, old_centroids):
            break

    return labels, centroids


#  K-MEDIAN MANUAL 
def kmedian_manual(data, k, max_iter=100):
    np.random.seed(42)
    data_np = data.to_numpy()
    centroids = data.sample(n=k).to_numpy()
    
    for _ in range(max_iter):
        distances = np.zeros((data_np.shape[0], k))
        for i in range(data_np.shape[0]):
            for j in range(k):
                distances[i, j] = np.sum(np.abs(data_np[i] - centroids[j]))
        
        labels = np.argmin(distances, axis=1)
        old_centroids = centroids.copy()
        centroids = np.array([np.median(data_np[labels == i], axis=0) for i in range(k)])
        
        if np.allclose(centroids, old_centroids):
            break

    return labels, centroids


# CLARA MANUAL 
def clara_manual(data, k, sample_size=None, max_iter=5, inner_max_iter=100):
    """
    Implementasi sederhana algoritma CLARA (Clustering Large Applications)
    Menggunakan K-Medoids pada sampel data untuk mencari representasi terbaik.
    """
    np.random.seed(42)
    data_np = data.to_numpy()
    n_samples = len(data_np)
    
    if sample_size is None:
        sample_size = min(40 + 2 * k, n_samples)

    best_medoids = None
    best_labels = None
    best_cost = np.inf

    for _ in range(max_iter):
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        sample_data = data_np[sample_idx]

        # Inisialisasi medoid secara acak
        medoids_idx = np.random.choice(sample_size, k, replace=False)
        medoids = sample_data[medoids_idx]

        for _ in range(inner_max_iter):
            # Hitung jarak Manhattan
            distances = np.zeros((sample_size, k))
            for i in range(sample_size):
                for j in range(k):
                    distances[i, j] = np.sum(np.abs(sample_data[i] - medoids[j]))

            labels = np.argmin(distances, axis=1)
            old_medoids = medoids.copy()

            # Update medoid tiap cluster
            for cluster_id in range(k):
                cluster_points = sample_data[labels == cluster_id]
                if len(cluster_points) == 0:
                    continue
                medoid_idx = np.argmin(
                    np.sum(np.abs(cluster_points[:, None] - cluster_points), axis=(1, 2))
                )
                medoids[cluster_id] = cluster_points[medoid_idx]

            if np.allclose(medoids, old_medoids):
                break

        # Evaluasi total cost pada seluruh data
        full_distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                full_distances[i, j] = np.sum(np.abs(data_np[i] - medoids[j]))
        full_labels = np.argmin(full_distances, axis=1)
        total_cost = np.sum(np.min(full_distances, axis=1))

        if total_cost < best_cost:
            best_cost = total_cost
            best_medoids = medoids
            best_labels = full_labels

    return best_labels, best_medoids
