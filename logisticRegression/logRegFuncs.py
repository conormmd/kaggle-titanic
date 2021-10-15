import numpy as np

def sigmoid(z):
    g = np.zeros_like(z)
    g = 1/ (1 + np.exp(-z))
    return g

def costFunction(theta, X, y, lambda_):
    m = len(y)  # number of training examples
    J = 0
    grad = np.zeros_like(theta)
    h = sigmoid(X @ theta.T)

    reg = theta  #regularisation initilised to theta (to then minimise)
    reg[0] = 0       #preventing bias term from being regularised

    J = (1 / m) * np.sum(-y @ np.log(h) - (1 - y) @ np.log(1 - h))
    J = J + (lambda_/(2*m)) * np.sum( np.square(reg) )
    grad = (1 / m) * ((h - y) @ X)
    grad = grad + (lambda_/m) * reg
    return J, grad

def predict(theta,X):
    m,n = np.shape(X)
    p = np.zeros(m)
    p = np.round( sigmoid(X @ theta.T) )
    return p