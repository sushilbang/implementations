import numpy as np

class LinearRegression:
    #constructor
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # m x n 
        self.weights = np.zeros(n_features) # n x 1
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias # m x n * n x 1 = m x 1

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) # n x m * m x 1 = n x 1
            db = (1/n_samples) * np.sum(y_pred - y) # 1 x 1

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
