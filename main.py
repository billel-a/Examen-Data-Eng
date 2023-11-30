from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pandas as pd
import plotly.express as px
import pandas as pd
import prince
from sklearn.manifold import TSNE


from sklearn.manifold import LocallyLinearEmbedding

def dim_red(mat, p , method):
  if (method == "ACP"):
    df = pd.DataFrame(mat)
    pca  = prince.PCA(n_components=p)
    red_mat = pca.fit_transform(df)
    red_mat = np.array(red_mat)
    return red_mat
  if (method == "TSNE"):
    model = TSNE(n_components = p )
    red_mat = model.fit_transform(mat)
    return red_mat
  elif(method=="LLE"):
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=p)
    red_mat = lle.fit_transform(mat)
    return red_mat
  else:
    raise Exception("Please select one of the three methods : APC, AFC, UMAP, LLE")

def clust_kmeans(mat,k):
  kmeans = KMeans(n_clusters=k,n_init=15)
  kmeans.fit(mat)
  return kmeans.predict(mat)

def clust_AgglomerativeClustering(mat,k):
  hierarchical = AgglomerativeClustering(n_clusters=k)
  pred = hierarchical.fit_predict(mat)
  return pred

def clust_DBSCAN(mat,k):
  dbscan = DBSCAN(eps=0.5, min_samples=k)
  pred = dbscan.fit_predict(mat)
  return pred

def clust(mat, k,method = "KMeans"):
  # Apply K-means clustering
  if (method == "KMeans"):
    return clust_kmeans(mat,k)
  elif (method == "Agglomerative"):
    return clust_AgglomerativeClustering(mat,k)
  elif (method == "DBScan"):
    return clust_DBSCAN(mat,k)
  else:
    raise Exception("Please select one of the three methods : KMeans, agglomerative, DBScan")

import pickle
with open("data.pkl" , "rb") as file:
  embeddings,labels = pickle.load(file)

# import data
k = len(set(labels))

# embedding
method = "ACP"
# perform dimentionality reduction
red_emb = dim_red(embeddings, 20,method )

# perform clustering
pred = clust(red_emb, k)

# evaluate clustering results
nmi_score = normalized_mutual_info_score(pred,labels)
ari_score = adjusted_rand_score(pred,labels)

print(f'NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')

import warnings
# Fit clustering 20 times
nmi_scores = []
ari_scores = []
n_iterations = 3
# Suppress FutureWarning related to KMeans
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
for _ in range(n_iterations):
  pred = clust(red_emb, k)
  # Evaluate clustering results
  nmi_score = normalized_mutual_info_score(pred, labels)
  ari_score = adjusted_rand_score(pred, labels)

  # Append scores for mean calculation
  nmi_scores.append(nmi_score)
  ari_scores.append(ari_score)

# Compute mean scores
mean_nmi = np.mean(nmi_scores)
mean_ari = np.mean(ari_scores)

# Print mean scores
print(f'NMI Scores Mean: {np.mean(nmi_scores):.2f} +/- {np.std(nmi_scores):.2f}')
print(f'ARI Scores Mean: {np.mean(ari_scores):.2f} +/- {np.std(ari_scores):.2f}')


