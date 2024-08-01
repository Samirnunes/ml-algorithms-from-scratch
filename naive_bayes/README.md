# Gaussian Naive Bayes Algorithm

Naive Bayes is a simple method for constructing classifiers. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all Naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable [1]. 

Here, I implement the Gaussian Naive Bayes Algorithm, which consist in the hypothesis that the probability density of a continuous feature given a class of the target (i.e. the likelihood function of that feature for a given target class) is a normal distribution with mean &mu;<sub>k</sub> and standard deviation &sigma;<sub>k</sub>, that is, mean and standard deviation of the values associated with the class C<sub>k</sub>. 

This algorithm can be only used with continuous variables to make sense. Otherwise, if you have only discrete features, you may use the Multinomial Naive Bayes algorithm. However, if you have both types of variables, you need, for example, to fit a Gaussian for the continuous ones and a Multinomial to the discrete ones, and them create another model to make the prediction considering the output probabilities from both models as features.

## Implementation

The implementation is based in the mathematical theory involving the Gaussian Naive Bayes Algorithm, which considers the use of the Bayes' Theorem for PDFs (Probability Density Functions) to obtain the posterior probability function. For the decision about the class, it's considered the Maximum a Posteriori (MAP) decision rule, which chooses the class that maximizes the posterior probability function. All the mathematical development can be seen in the reference [1], which can be found in the `references` folder. All the implementation is present in the `gaussian_naive_bayes.py` file, in the `src` folder.

### Tools

- Python
- Pandas
- Numpy
- Jupyter Notebook

## Data

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/linear_regression/images/data.png" alt="Material Bread logo">
<p>

## Results

The file `test.ipynb`, in the `src` folder , contains the predictions in the in the university admission test data given by the models trained in the train data, so it can be showed that the result of the implemented algorithm is the same of the algorithm present in the `scikit-learn` library [2] without the use of variable smoothing.

- Predictions of the implemented algorithm:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/naive_bayes/images/prediction_implemented_model.png" alt="Material Bread logo">
<p>

- Predictions of the algorithm implemented by `scikit-learn`:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/naive_bayes/images/prediction_sklearn_model.png" alt="Material Bread logo">
<p>

- Comparison

<p align="center">
    <img width="400" src="https://github.com/Samirnunes/ml-algorithms-from-scratch/blob/main/naive_bayes/images/comparison.png" alt="Material Bread logo">
<p>

It can be seen that the results are the same. Therefore, the implementation made in this repository is consistent with `scikit-learn`.

## References

[1] https://en.wikipedia.org/wiki/Naive_Bayes_classifier

[2] https://github.com/scikit-learn/scikit-learn/blob/70fdc843a/sklearn/naive_bayes.py#L147
