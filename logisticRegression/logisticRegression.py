import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from logRegFuncs import *

"""Reading & Sorting Data"""
data = pd.read_csv("./data/train.csv")

y = data.Survived.to_numpy()
data = data.drop("Survived", axis=1)
data = data.drop("PassengerId", axis = 1)

X = data.to_numpy()
m,n = np.shape(X)
X = np.c_[np.ones(m), X]

"""Initialising Parameter Search"""
theta = trainModel(X, y)

"""Reading & Sorting CV Data"""
data = pd.read_csv("./data/CV.csv")

y_CV = data.Survived.to_numpy()
data = data.drop("Survived", axis=1)
data = data.drop("PassengerId", axis = 1)

X_CV = data.to_numpy()
m,n = np.shape(X_CV)
X_CV = np.c_[np.ones(m), X_CV]

p = predict(theta, X_CV)
print("Basic model accuracy = ", np.mean(p == y_CV))

"""Plotting CV Error Against Range of Lambdas to Optimise"""
lambdas = np.array([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])

error_train, error_CV = validationCurve(X, y, X_CV, y_CV, lambdas)

lambda_errors = pd.DataFrame()
lambda_errors["Lambda"] = lambdas
lambda_errors["Train Error"] = error_train
lambda_errors["CV Error"] = error_CV

plt.loglog(lambdas, error_train, label = "Train"); plt.loglog(lambdas, error_CV, label = "CV"); plt.legend(); plt.xlabel("Lambdas"); plt.ylabel("Errors"); plt.title("Validation Curve"); plt.show()
print(lambda_errors)

"""Using Optimal Lambda To Output Thetas"""
np.random.seed()

lambda_optimal = 10
theta = trainModel(X, y, lambda_ = lambda_optimal)

p = predict(theta, X_CV)
print("Optimised model on CV accuracy = ",np.mean(p == y_CV))
print("Optimised theta values = ", theta)

"""Generating CV & Train Learning Curves"""
error_train, error_CV = learningCurve(X, y, X_CV, y_CV, lambda_ = lambda_optimal)
num_data = np.arange(1, y.size+1)

learning_curve = pd.DataFrame()
learning_curve["Data Quantity"] = num_data
learning_curve["Train Error"] = error_train
learning_curve["CV Error"] = error_CV

print(learning_curve)
plt.plot(num_data, error_train, label="Train"); plt.plot(num_data, error_CV, label = "CV"); plt.legend(); plt.xlabel("Num of Data"); plt.ylabel("Error"); plt.title("Learning Curve"); plt.show()


"""Generating Test Results"""
test = pd.read_csv('./data/test.csv')
passengerId = test.PassengerId
test = test.drop("PassengerId", axis=1)

X_test = test.to_numpy()
m,n = np.shape(X_test)
X_test = np.c_[np.ones(m), X_test]

y_predicted = predict(theta, X_test)

solution = pd.DataFrame()
solution["PassengerId"] = passengerId
solution["Survived"] = y_predicted
solution.Survived = solution.Survived.astype(int)
solution.to_csv("./logisticRegression/solution.csv", sep=",", index=False)


