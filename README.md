# Classification of the NG20 dataset using dimensionality reduction and clustering

## Project Objective:
The project aims to develop a clustering model using dimensionality reduction through PCA, T-SNE, UMAP and LLE. The process will be carried out sequentially, 
combining each dimensionality reduction method with clustering algorithm like Kmeans, Agglomerative and DBScan. 
The results will be evaluated using NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index).

## Code :
This is the source code for the [classification of the NG20 dataset](https://github.com/billel-a/Examen-Data-Eng).

## Solutuoin Overview
![image](https://github.com/billel-a/Examen-Data-Eng/blob/main/shema_project.png)

## Objectives

✅ Developing the model locally with Docker and mounting a volume on the project (docker run -v C:\Users\MLDSadmin\Exam_V2\Examen-Data-Eng --name container_exam akninebillel/imageexamen:1).

✅ Visualizing data on a 2D and 3D plane using PCA, MCA, and UMAP.

✅ Writing good documentation (README).

✅ Saving data to avoid downloading it (pickle file) with each instantiation of a new container (or installing it in the Docker image).

✅ Performing different initializations and displaying the mean (and standard deviation) of the results.

✅ Adding dimensionality reduction method (LLE) and clustering algorithm (Agglomerative and DBScan).

#### Dependencies
- scikit-learn==0.24.2
- numpy
- pandas
- prince
- umap-learn
- SentenceTransformer


## Final Best tandem approch 
The most effective combination was achieved by integrating UMAP with K-means, resulting in the highest NMI of 0.49 and ARI of 0.3.
![image](https://github.com/billel-a/Examen-Data-Eng/blob/main/NMI_ARI_UMAP_Kmeans.png)

