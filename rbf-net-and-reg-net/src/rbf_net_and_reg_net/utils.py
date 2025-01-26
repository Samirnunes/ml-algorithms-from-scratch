import pandas as pd
from sklearn.model_selection import train_test_split


def import_data():
    data = pd.read_csv("../../data/data.csv")
    data = data.replace(
        {"n": "0", "y": "1", "democrat": "0", "republican": "1"}
    ).astype(int)
    X = data.drop(["Alvo"], axis=1)
    y = data["Alvo"]
    return _split_data(X, y)


def _split_data(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train.values, X_test.values, y_train.values, y_test.values
