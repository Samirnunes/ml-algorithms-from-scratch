from lab5.logger import logger
from lab5.rbf_net import RBFNetwork
from lab5.reg_net import NoRegularizationNetwork, RegularizationNetwork
from sklearn.metrics import accuracy_score
from utils import import_data

RANDOM_STATE = 0


def a():
    logger.info("\na)")

    X_train, X_test, y_train, y_test = import_data()
    reg_net = NoRegularizationNetwork(RANDOM_STATE)
    reg_net.fit(X_train, y_train)

    logger.info(
        f'"Regularization" network with no regularization\nNetwork weights:\n{reg_net.weights}'
    )

    if reg_net.fitted:
        y_train_pred = reg_net.predict(X_train)
        logger.info(f"Train set accuracy: {accuracy_score(y_train, y_train_pred)}")

        y_test_pred = reg_net.predict(X_test)
        logger.info(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)}\n")
    else:
        logger.info("")

def b():
    logger.info("\nb)")

    X_train, X_test, y_train, y_test = import_data()
    lambda_values = [0, 1, 10, 100, 1000]

    for lambda_param in lambda_values:
        reg_net = RegularizationNetwork(lambda_param, RANDOM_STATE)
        reg_net.fit(X_train, y_train)
        logger.info(
            f"Regularization network with lambda = {lambda_param}\nNetwork weights:\n{reg_net.weights}"
        )

        if reg_net.fitted:
            y_train_pred = reg_net.predict(X_train)
            logger.info(f"Train set accuracy: {accuracy_score(y_train, y_train_pred)}")

            y_test_pred = reg_net.predict(X_test)
            logger.info(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)}\n")
        else:
            logger.info("")

def c():
    logger.info("\nc)")

    X_train, X_test, y_train, y_test = import_data()
    n_neurons = 14
    rbf_net = RBFNetwork(n_neurons, RANDOM_STATE)
    rbf_net.fit(X_train, y_train)

    logger.info(
        f"RBF network with {n_neurons} neurons\nNetwork weights:\n{rbf_net.weights}"
    )

    y_train_pred = rbf_net.predict(X_train)
    logger.info(f"Train set accuracy: {accuracy_score(y_train, y_train_pred)}")

    y_test_pred = rbf_net.predict(X_test)
    logger.info(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)}\n")


def d():
    logger.info("\nd)")

    X_train, X_test, y_train, y_test = import_data()
    n_neurons = 25
    rbf_net = RBFNetwork(n_neurons, RANDOM_STATE)
    rbf_net.fit(X_train, y_train)

    logger.info(
        f"RBF network with {n_neurons} neurons\nNetwork weights:\n{rbf_net.weights}"
    )

    y_train_pred = rbf_net.predict(X_train)
    logger.info(f"Train set accuracy: {accuracy_score(y_train, y_train_pred)}")

    y_test_pred = rbf_net.predict(X_test)
    logger.info(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)}\n")


def e():
    logger.info("\ne)")

    X_train, X_test, y_train, y_test = import_data()
    n_neurons = 50
    rbf_net = RBFNetwork(n_neurons, RANDOM_STATE)
    rbf_net.fit(X_train, y_train)

    logger.info(
        f"RBF network with {n_neurons} neurons\nNetwork weights:\n{rbf_net.weights}"
    )

    y_train_pred = rbf_net.predict(X_train)
    logger.info(f"Train set accuracy: {accuracy_score(y_train, y_train_pred)}")

    y_test_pred = rbf_net.predict(X_test)
    logger.info(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)}\n")


def f():
    logger.info("\nf)")

    X_train, X_test, y_train, y_test = import_data()
    n_neurons = 80
    rbf_net = RBFNetwork(n_neurons, RANDOM_STATE)
    rbf_net.fit(X_train, y_train)

    logger.info(
        f"RBF network with {n_neurons} neurons\nNetwork weigths:\n{rbf_net.weights}"
    )

    y_train_pred = rbf_net.predict(X_train)
    logger.info(f"Train set accuracy: {accuracy_score(y_train, y_train_pred)}")

    y_test_pred = rbf_net.predict(X_test)
    logger.info(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)}\n")
