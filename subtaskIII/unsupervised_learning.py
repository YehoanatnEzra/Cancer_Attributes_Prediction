import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

from process_data import remove_cols, clean_data

# Load and preprocess data
train_path = "../data/train_test_splits/train.feats.csv"
df = pd.read_csv(train_path, dtype=str, low_memory=False)
df_clean = clean_data(remove_cols(df))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df_clean)

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# --- PCA ---
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("# Components")
plt.ylabel("Explained Variance")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pca_explained_variance.png"))
plt.close()

# Visualize first 2 components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title("PCA - First 2 Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(output_dir, "pca_2d_projection.png"))
plt.close()

# --- KMeans Clustering ---
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(X)
sil_score = silhouette_score(X, clusters)
print(f"KMeans (k={k}) Silhouette Score: {sil_score:.4f}")

# Plot PCA with cluster coloring
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", legend="full")
plt.title("PCA + KMeans Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(output_dir, "pca_kmeans_clusters.png"))
plt.close()

# # --- t-SNE Visualization ---
# sample_size = 2000  # or 1000
# if X.shape[0] > sample_size:
#     indices = np.random.choice(X.shape[0], sample_size, replace=False)
#     X_sampled = X[indices]
#     clusters_sampled = clusters[indices]
# else:
#     X_sampled = X
#     clusters_sampled = clusters
#
# tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(X)
#
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette="tab10", legend="full")
# plt.title("t-SNE with KMeans Clusters")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.savefig(os.path.join(output_dir, "tsne_kmeans_clusters.png"))
# plt.close()

