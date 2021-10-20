import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from logRegFuncs import *

data = pd.read_csv("./data/train.csv")

y = data.Survived.to_numpy()
data = data.drop("Survived", axis=1)
data = data.drop("PassengerId", axis = 1)

X = data.to_numpy()
m,n = np.shape(X)
X = np.c_[np.ones(m), X]

theta = trainModel(X, y)
print(theta)