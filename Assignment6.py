# Practical Assignment 6: Clustering Algorithms

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load the Wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
actual_labels = wine.target

print("Dataset Shape:", X.shape)
print("\nFirst five records:")
print(X.head())
print("\nMissing values:", X.isnull().sum().sum())

# 2. Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 4. Apply Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)

# 5. Apply Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3, random_state=42, n_init=5
)
gmm_labels = gmm.fit_predict(X_scaled)

# 6. Apply DBSCAN clustering
dbscan = DBSCAN(eps=2.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 7. Store all clustering results
models = {
    "K-Means": kmeans_labels,
    "Agglomerative": agg_labels,
    "Gaussian Mixture": gmm_labels,
    "DBSCAN": dbscan_labels
}

# 8. Evaluate clustering algorithms
results = []

for name, labels in models.items():

    # DBSCAN uses -1 to identify noise points
    if name == "DBSCAN":
        valid = labels != -1
        metric_labels = labels[valid]
        metric_data = X_scaled[valid]
        noise = sum(labels == -1)
    else:
        metric_labels = labels
        metric_data = X_scaled
        noise = 0

    clusters = len(set(metric_labels))

    silhouette = None
    db_index = None
    ch_score = None

    # Calculate metrics when at least two clusters are available
    if 2 <= clusters < len(metric_labels):
        silhouette = silhouette_score(
            metric_data, metric_labels
        )
        db_index = davies_bouldin_score(
            metric_data, metric_labels
        )
        ch_score = calinski_harabasz_score(
            metric_data, metric_labels
        )

    # Actual labels are used only for external evaluation
    ari = adjusted_rand_score(actual_labels, labels)

    results.append({
        "Algorithm": name,
        "Clusters": clusters,
        "Noise Points": noise,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Score": ch_score,
        "Adjusted Rand Index": ari
    })

results_df = pd.DataFrame(results)

print("\nClustering Evaluation Results:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("clustering_results.csv", index=False)

# 9. Reduce dimensions to visualize clusters
# PCA is used only for visualization, not for clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (name, labels) in zip(axes.flatten(), models.items()):
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="viridis",
        s=45
    )
    ax.set_title(name + " Clusters")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

plt.tight_layout()
plt.savefig("cluster_comparison.png", dpi=300)
plt.show()

# 10. Visualize actual wine groups for reference
plt.figure(figsize=(7, 5))

for i, name in enumerate(wine.target_names):
    mask = actual_labels == i
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=name,
        s=45
    )

plt.title("Wine Groups (Reference Labels)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("wine_reference_groups.png", dpi=300)
plt.show()

# 11. Draw the hierarchical clustering dendrogram
tree = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(tree, no_labels=True)
plt.title("Agglomerative Clustering Dendrogram")
plt.xlabel("Wine Samples")
plt.ylabel("Linkage Distance")
plt.tight_layout()
plt.savefig("agglomerative_dendrogram.png", dpi=300)
plt.show()

# 12. Compare K-Means using different cluster counts
experiment = []

for k in range(2, 7):
    model = KMeans(
        n_clusters=k, random_state=42, n_init=10
    )
    labels = model.fit_predict(X_scaled)

    experiment.append({
        "Number of Clusters": k,
        "Silhouette Score": silhouette_score(
            X_scaled, labels
        ),
        "Davies-Bouldin Index": davies_bouldin_score(
            X_scaled, labels
        ),
        "Calinski-Harabasz Score": calinski_harabasz_score(
            X_scaled, labels
        )
    })

experiment_df = pd.DataFrame(experiment)

print("\nK-Means Cluster Count Comparison:")
print(experiment_df.round(4).to_string(index=False))

experiment_df.to_csv(
    "kmeans_cluster_experiment.csv", index=False
)

# 13. Compare algorithms using actual calculated scores
valid_results = results_df.dropna(
    subset=["Silhouette Score", "Davies-Bouldin Index"]
)

if not valid_results.empty:
    best_silhouette = valid_results.loc[
        valid_results["Silhouette Score"].idxmax()
    ]
    best_db = valid_results.loc[
        valid_results["Davies-Bouldin Index"].idxmin()
    ]

    print("\nConclusion:")
    print(
        "Highest Silhouette Score:",
        best_silhouette["Algorithm"]
    )
    print(
        "Lowest Davies-Bouldin Index:",
        best_db["Algorithm"]
    )
    print("Higher Silhouette Score is generally better.")
    print("Lower Davies-Bouldin Index is generally better.")
    print("Compare multiple metrics before selecting a model.")

