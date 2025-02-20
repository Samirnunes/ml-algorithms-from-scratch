# Linear Regression with Mini-Batches Stochastic Gradient Descent

- Article on Medium: https://medium.com/@samir.silva12342/implementação-de-modelos-do-zero-regressão-linear-b3cdb9de1a24
 
The linear regression is one of the most well-known models in the context of machine learning, involving fitting a linear model to data whose predictors (features) are highly correlated with the target value to be estimated. It can be implemented through the Stochastic Gradient Descent algorithm: an iterative optimization algorithm for the parameters of the regression function based on minimizing the cost (or loss) function. This minimization is achieved by calculating the negative gradient of the cost function with respect to the parameters of the regression function (weights and bias).

The codes step-by-step develop the linear regression model, following a logical progression for learning its application and implementation. They adhere to best practices commonly used in machine learning.

## Implementation

The implementation is present in the file `linear_regressor.py`, in the `src` folder.

<p align="center">
    <img width="800" src="./images/explanation1.png" alt="Material Bread logo">
<p>

<p align="center">
    <img width="800" src="./images/explanation2.png" alt="Material Bread logo">
<p>

<p align="center">
    <img width="800" src="./images/explanation3.png" alt="Material Bread logo">
<p>

### Tools

- Jupyter Notebook
- Python
- Pandas
- Numpy
- Matplotlib

## Data

Data for Admission in the University from Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?source=post_page-----b3cdb9de1a24--------------------------------

<p align="center">
    <img width="800" src="./images/data.png" alt="Material Bread logo">
<p>

## Results

- Correlations Heatmap

<p align="center">
    <img width="600" src="./images/correlations_heatmap.png" alt="Material Bread logo">
<p>

- Model's Weights X Regularization Parameter

<p align="center">
    <img width="600" src="./images/weights_lambda.png" alt="Material Bread logo">
<p>

- Train Loss

 <p align="center">
    <img width="600" src="./images/train_loss.png" alt="Material Bread logo">
<p>

- Final Test Loss

<p align="center">
    <img width="400" src="./images/final_loss.png" alt="Material Bread logo">
<p>
