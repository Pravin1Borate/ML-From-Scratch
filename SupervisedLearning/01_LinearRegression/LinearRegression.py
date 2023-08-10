import numpy as np

class LinearRegression:

    def __init__(self,learning_rate=0.01,n_iterations=1000,) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = None
        self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X,self.weight) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self,X):
        y_pred = np.dot(X,self.weight) + self.bias
        return y_pred
