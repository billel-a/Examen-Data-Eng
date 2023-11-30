# Classification of the NG20 dataset using dimensionality reduction and clustering

## Project Objective:
The project aims to develop a clustering model using dimensionality reduction through PCA, T-SNE, UMAP and LLE. The process will be carried out sequentially, 
combining each dimensionality reduction method with clustering algorithm like Kmeans, Agglomerative and DBScan. 
The results will be evaluated using NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index).

## Code :
This is the source code for the [classification of the NG20 dataset](https://github.com/billel-a/Examen-Data-Eng).

## Solutuoin Overview
![image](https://github.com/billel-a/Examen-Data-Eng/blob/main/shema_project.png)

#### Dependencies
- scikit-learn==0.24.2
- numpy
- pandas
- prince
- umap-learn
- SentenceTransformer


## Final Best tandem approch 
The most effective combination was achieved by integrating T-SNE with K-means, resulting in the highest NMI of 0.41 and ARI of 0.247.


