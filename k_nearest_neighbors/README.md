# K-Nearest Neighbors Algorithm

The K-Nearest Neighbors Algorithm (k-NN) is a non-parametric supervised learning method used for classification and regression. The training examples are vectors in a multidimensional feature space, each with a class label. The training phase of the algorithm consists only of storing the feature vectors and class labels of the training samples. Besides, in the classification phase, k is a user-defined constant, and an unlabeled vector (a test instance) is classified by assigning the label which is most frequent among the k training samples nearest to that point, in the case of classification, or the value which is the mean of the labels of the k nearest neighbors of that point, in the case of regression [1]. In this work, the Euclidean distance is used as distance metric.

## Data

- Classification

Binary Classification Bank Churn Dataset Cleaned from Kaggle: https://www.kaggle.com/datasets/prishasawhney/binary-classification-bank-churn-dataset-cleaned?select=train_cleaned.csv

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/decision_tree/images/data_image.png" alt="Material Bread logo">
<p>

- Regression

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/linear_regression/images/data.png" alt="Material Bread logo">
<p>

## Implementation

All the implementation is present in the file `k_nearest_neighbors.py`. The theory used for that is based in the reference [1].

## Results

For results, we will be comparing the output of the implemented algorithm and that from the `scikit-learn` library.



## References

[1] https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
