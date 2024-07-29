# Gaussian Naive Bayes Algorithm

Naive Bayes is a simple method for constructing classifiers. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all Naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. [1]

Here, I implement the Gaussian Naive Bayes Algorithm, which consist in the hypothesis that the probability density of a feature given a class of the target (i.e. the likelihood function of that feature for a given target class) is a normal distribution with mean &mu;<sub>k</sub> and standard deviation &sigma;<sub>k</sub>, that is, mean and standard deviation of the values associated with the class C<sub>k</sub>.

## Data

Binary Classification Bank Churn Dataset Cleaned from Kaggle: https://www.kaggle.com/datasets/prishasawhney/binary-classification-bank-churn-dataset-cleaned?select=train_cleaned.csv

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/decision_tree/images/data_image.png" alt="Material Bread logo">
<p>

## Implementation

The implementation is based in the mathematical theory involving the Gaussian Naive Bayes Algorithm, which considers the use of the Bayes' Theorem for PDFs (Probability Density Functions) to obtain the posterior probability function. For the decision about the class, it's considered the Maximum a Posteriori (MAP) decision rule, which chooses the class that maximizes the posterior probability function. All the mathematical development can be seen in the reference [1], which can be found in the folder `references`. All the implementation is present in the file `gaussian_naive_bayes.py`, in the folder `src`.

## Results

The file `test.ipynb`, in the folder `src`, contains the predictions in the in the bank churn test data given by the models trained in the train data, so it can be showed that the result of the implemented algorithm is the same of the algorithm present in the `scikit-learn` library [2].

- Predictions of the implemented algorithm:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/naive_bayes/images/probability_implemented_model.png" alt="Material Bread logo">
<p>

- Predictions of the algorithm implemented by `scikit-learn`:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/naive_bayes/images/probability_sklearn_model.png" alt="Material Bread logo">
<p>

## References

[1] https://en.wikipedia.org/wiki/Naive_Bayes_classifier

[2] https://github.com/scikit-learn/scikit-learn/blob/70fdc843a/sklearn/naive_bayes.py#L147
