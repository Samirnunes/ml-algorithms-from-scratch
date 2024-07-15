from parameters import Parameters


class LinearRegressionParameters(Parameters):
    def __init__(self):
        self.initial_weights = []
        self.initial_bias = 0
        self.epochs = 100
        self.batch_size = 10
        self.alpha = 0.1
        self.lambda_reg = 0.1  # L2 regularization
        self.gamma_reg = 0.0  # L1 regularization
        self.random_state = 0
