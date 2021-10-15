import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from logRegFuncs import *

"""Reading & Sorting Data"""
data = pd.read_csv("./data/train.csv")

y = data.Survived.to_numpy()
data = data.drop("Survived", axis=1)

X = data.to_numpy()
m,n = np.shape(X)
X = np.c_[np.ones(m), X]

"""Initialising Parameter Search"""
thetaInit = np.zeros(X.shape[1])

lambda_ = 10
options = {'maxiter':10}
res = optimize.minimize(costFunction,       #Our cost function being used
                        thetaInit,          #Initial values for theta
                        (X, y, lambda_),    #Data being fitted against
                        jac=True,           #Return jacobian (gradient)
                        method='TNC',       #Truncated Newton algorithm
                        options=options)    #Using our chosen number of max iterations

theta=res.x; cost=res.fun


"""Reading & Sorting CV Data"""
data = pd.read_csv("./data/CV.csv")

y_CV = data.Survived.to_numpy()
data = data.drop("Survived", axis=1)

X_CV = data.to_numpy()
m,n = np.shape(X_CV)
X_CV = np.c_[np.ones(m), X_CV]

p = predict(theta, X_CV)
print(np.mean(p == y_CV))

lambdas = np.array([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
n = len(lambdas)
error = np.zeros(n)
repeats = 200

for j in range(repeats):
    np.random.seed()
    for i in range(n):
        thetaInit = np.random.rand(X.shape[1])/100
        lambda_ = lambdas[i]
        options = {'maxiter':400}
        res = optimize.minimize(costFunction,       #Our cost function being used
                                thetaInit,          #Initial values for theta
                                (X, y, lambda_),    #Data being fitted against
                                jac=True,           #Return jacobian (gradient)
                                method='TNC',       #Truncated Newton algorithm
                                options=options)    #Using our chosen number of max iterations
        theta = res.x; cost = res.fun
        p = predict(theta, X_CV)
        error[i] = error[i] + (1 - np.mean(p == y_CV))
    print(error)

error = error/repeats
plt.loglog(lambdas, error)
print(error)
plt.show()

