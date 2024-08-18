import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

class Perceptron:
    def __init__(self, eta=0.5, max_iter=1000, problem_type='classification', plot=False):
        self.eta = eta  # Learning rate
        self.max_iter = max_iter  # Maximum number of iterations
        self.plot = plot  # Control whether to plot during training
        self.problem_type = problem_type  # 'classification' or 'regression'
        self.weights = None  # Placeholder for weights

    def fit(self, X, Y):
        self.weights = np.zeros(X.shape[1])
        print(f"The Learning Rate is {self.eta}")
        print(f"The initial Weights are {self.weights}")

        for epoch in range(1, self.max_iter + 1):
            w_old = self.weights.copy()
            print(f"Running Epoch {epoch} ============>")
            total_grad = np.zeros_like(self.weights)
            for i in range(len(X)):
                if self.problem_type == 'classification':
                    yhat = 1 if np.dot(X[i], w_old) > 0 else 0
                    if yhat == Y[i]:
                        continue
                    elif yhat == 1 and Y[i] == 0:
                        self.weights -= self.eta * X[i]
                    elif yhat == 0 and Y[i] == 1:
                        self.weights += self.eta * X[i]
                elif self.problem_type == 'regression':
                    yhat = np.dot(X[i], w_old)
                    gradient = (Y[i] - yhat) * X[i]
                    total_grad += gradient
            if self.problem_type == 'regression':
                self.weights += self.eta * total_grad
            
            print(f"Old Weights {w_old}")
            print(f"New Weights: {self.weights if not np.allclose(w_old, self.weights) else 'None'}")
            
            if self.plot and self.problem_type == 'classification':
                self.plot_graph(X, Y)
            
            if np.allclose(w_old, self.weights):
                print(f"Normal Vector of Hyperplane = {self.weights}")
                break

    def plot_graph(self, X, Y):
        if self.problem_type == 'classification':
            w0, w1, w2 = self.weights
            x1 = np.linspace(-2, 2, 100)
            x2 = (-w0 - w1 * x1) / w2

            plt.scatter(X[:, 1], X[:, 2], c=Y, cmap='coolwarm')
            plt.plot(x1, x2, '-r')
            plt.legend(['Datapoints', 'Hyperplane - (wTx = 0)'])
            plt.show()

    def predict(self, X):
        if self.problem_type == 'classification':
            return np.where(np.dot(X, self.weights) > 0, 1, 0)
        elif self.problem_type == 'regression':
            return np.dot(X, self.weights)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        if self.problem_type == 'classification':
            accuracy = accuracy_score(Y, predictions)
            report = classification_report(Y, predictions)
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
        elif self.problem_type == 'regression':
            mse = mean_squared_error(Y, predictions)
            print(f"Mean Squared Error: {mse}")

    # Example usage:
    # Sample dataset
    df = pd.DataFrame(data=[[0,0,0],[0,1,0],[1,0,0],[1,1,1]], columns=['x1','x2','y'])
    df['x0'] = 1  # Adding bias term
    
    X = np.array(df[['x0','x1','x2']])
    Y = np.array(df['y'])
    
    # Classification
    # perceptron_cls = Perceptron(eta=1.0, max_iter=100, problem_type='classification', plot=True)
    # perceptron_cls.fit(X, Y)
    # perceptron_cls.evaluate(X, Y)
    
    # # Regression (using slightly different data for regression)
    # df_reg = pd.DataFrame(data=[[0,0,0],[0,1,1],[1,0,1],[1,1,2]], columns=['x1','x2','y'])
    # df_reg['x0'] = 1  # Adding bias term
    
    # X_reg = np.array(df_reg[['x0','x1']])
    # Y_reg = np.array(df_reg['y'])
    
    # # Regression
    # perceptron_reg = Perceptron(eta=0.001, max_iter=1000, problem_type='regression')
    # perceptron_reg.fit(X_reg, Y_reg)
    # perceptron_reg.evaluate(X_reg, Y_reg)
