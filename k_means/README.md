# K-means with Lloyd's Algorithm



## Data

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/linear_regression/images/data.png" alt="Material Bread logo">
<p>

## Implementation

## Results

### Two features (GRE Score and CGPA)

- Implementation

Count of labels after clustering

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_count_implementation.png" alt="Material Bread logo">
<p>

WCSS along iterations

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_wcss_implementation.png" alt="Material Bread logo">
<p>

Scatter plot with clusters

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_clusters_implementation.png" alt="Material Bread logo">
<p>

- Scikit-learn

Count of labels after clustering

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_count_sklearn.png" alt="Material Bread logo">
<p>

Scatter plot with clusters

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/two_var_clusters_sklearn.png" alt="Material Bread logo">
<p>

We can see that the results with two variables are exactly the same.

### All features

- Implementation

Count of labels after clustering
  
<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_count_implementation.png" alt="Material Bread logo">
<p>

WCSS along iterations

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_wcss_implementationg.png" alt="Material Bread logo">
<p>

Centroids

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_centroids_implementation.png" alt="Material Bread logo">
<p>

- Scikit-learn

Count of labels after clustering

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_count_sklearn.png" alt="Material Bread logo">
<p>

Centroids

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/k_means/images/all_var_centroids_wcss_sklearn.png" alt="Material Bread logo">
<p>

With all the features, the results were almost equal, with feel differents due to the sensitivy of K-Means in respect to the initial centroids.

## References

[1] https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
