import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets

from LinearRegression import LinearRegression

class TestLinearRegression:
    def __init__(self,X,y) -> None:
        self.X = X
        self.y = y
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,\
                                                                             test_size=0.3,random_state=42)
        self.linear_regression = LinearRegression()

    def predictionLine(self):
        y_pred_line = self.linear_regression.predict(self.X)
        cmap = plt.get_cmap('viridis')
        fig = plt.figure(figsize=(8,6))
        m1 = plt.scatter(self.X_train, self.y_train, color=cmap(0.9), s=10)
        m2 = plt.scatter(self.X_test, self.y_test, color=cmap(0.5), s=10)
        plt.plot(self.X, y_pred_line, color='black', linewidth=2, label='Prediction')
        plt.savefig('Regression_Line.png')
        plt.show()

    def mean_squared_error(self,y_pred):
        mse = np.mean((self.y_test - y_pred)**2)
        return mse
    
    def RunRegression(self):
        self.linear_regression.fit(self.X_train,self.y_train)
        y_pred = self.linear_regression.predict(self.X_test)
        mse = self.mean_squared_error(y_pred=y_pred)
        print(f"Mean Squared Error is : {np.round(mse,2)}")

if __name__ == "__main__":
    X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=42)
    obj = TestLinearRegression(X,y)
    # obj.plot_data_points()
    obj.RunRegression()
    obj.predictionLine()


