import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from features_00 import load_and_extract_features
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
import hdbscan

images, names, X = load_and_extract_features()

results = {}

def evaluate(name, labels):
    valid = labels != -1 if -1 in labels else np.ones_like(labels, dtype=bool)

    if len(np.unique(labels[valid])) < 2:
        return {
            "clusters": len(set(labels) - {-1}),
            "silhouette": -1,
            "dbi": -1,
            "chi": -1,
            "noise": np.sum(labels == -1)
        }

    return {
        "clusters": len(set(labels) - {-1}),
        "silhouette": silhouette_score(X[valid], labels[valid]),
        "dbi": davies_bouldin_score(X[valid], labels[valid]),
        "chi": calinski_harabasz_score(X[valid], labels[valid]),
        "noise": np.sum(labels == -1)
    }

# ============================
# Run all models
# ============================

models = {
    "kmeans": KMeans(n_clusters=2, random_state=42).fit_predict(X),
    "agglomerative": AgglomerativeClustering(n_clusters=2).fit_predict(X),
    "spectral": SpectralClustering(n_clusters=2, affinity='nearest_neighbors').fit_predict(X),
    "dbscan": DBSCAN(eps=3, min_samples=3).fit_predict(X),
    "hdbscan": hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1).fit_predict(X)
}

for name, labels in models.items():
    results[name] = evaluate(name, labels)

# ============================
# Print comparison
# ============================

print("\n==============================")
print("     CLUSTERING COMPARISON")
print("==============================")

for name, metrics in results.items():
    print(f"\n{name.upper()}:")
    print(f"  Clusters:          {metrics['clusters']}")
    print(f"  Noise Points:      {metrics['noise']}")
    print(f"  Silhouette Score:  {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin:    {metrics['dbi']:.4f}")
    print(f"  Calinski-Harabasz: {metrics['chi']:.4f}")

# Rank by silhouette
ranked = sorted(results.items(), key=lambda x: x[1]['silhouette'], reverse=True)

print("\n\nBEST ALGORITHM:")
print(f" 🏆 {ranked[0][0].upper()} with Silhouette {ranked[0][1]['silhouette']:.4f}")
