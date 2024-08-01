# K-means with Lloyd's Algorithm

K-means is an unsupervised learning algorithm which aims to cluster data based in its features. The most common algorithm uses an iterative refinement technique, and its called Lloyd's Algorithm or, generally, the "k-means algorithm". It is sometimes also referred to as "naive k-means", because there exist faster alternatives. 

The algorithm is divided in two steps: assingment of each observation (row of the dataframe) to a cluster and centroids' update. The assingment is done by determining, for each observation, the nearest centroid and then assigning the observation to the cluster with that centroid. The nearest centroid is the one with the smallest squared Euclidean distance to the observation. In turn, the centroids' update is done for each cluster by calculating the mean of all the observations in the cluster and then assigning that mean as the new centroid. These steps are repeated until the centroids doesn't change between two iterations (with a tolerance) or the maximum number of iterations is reached.

## Data

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/linear_regression/images/data.png" alt="Material Bread logo">
<p>

## Implementation

The implementation is present in the file `k_means.py`, in the folder `src`. The theory used for that is based in the reference [1], which can be found in the `references` folder. For comparing the solution, you can see the implementation present in the `scikit-learn` library in reference [2].

### Tools

- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Results

### Two features (GRE Score and CGPA)

#### Implementation

- Count of labels after clustering

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_count_implementation.png" alt="Material Bread logo">
<p>

- WCSS along iterations

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_wcss_implementation.png" alt="Material Bread logo">
<p>

- Scatter plot with clusters

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_clusters_implementation.png" alt="Material Bread logo">
<p>

#### Scikit-learn

- Count of labels after clustering

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_count_sklearn.png" alt="Material Bread logo">
<p>

- Scatter plot with clusters

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_clusters_sklearn.png" alt="Material Bread logo">
<p>

We can see that the results with two variables are exactly the same.

### All features

#### Implementation

- Count of labels after clustering
  
<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_count_implementation.png" alt="Material Bread logo">
<p>

- WCSS along iterations

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_wcss_implementationg.png" alt="Material Bread logo">
<p>

- Centroids

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_centroids_implementation.png" alt="Material Bread logo">
<p>

#### Scikit-learn

- Count of labels after clustering

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_count_sklearn.png" alt="Material Bread logo">
<p>

- Centroids

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_centroids_wcss_sklearn.png" alt="Material Bread logo">
<p>

With all the features, the results were almost equal, with few differents due to the sensitivy of K-Means in respect to the initial centroids.

## References

[1] https://en.wikipedia.org/wiki/K-means_clustering

[2] https://github.com/scikit-learn/scikit-learn/blob/70fdc843a/sklearn/cluster/_kmeans.py#L1196
