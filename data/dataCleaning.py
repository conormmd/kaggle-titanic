import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import pandas as pd

data_raw = pd.read_csv('./data/raw_train.csv')
test_raw = pd.read_csv('./data/raw_test.csv')

data = data_raw.drop(["PassengerId","Name", "Ticket"], axis=1)
test = test_raw.drop(["Name", "Ticket"], axis=1)

#sex = {"male" : 1, "female" : 0}
#data.Sex = [sex[item] for item in data.Sex]

#embarked = {"Q" : 0, "C" : 1, "S" : 2}
#data.Embarked = [embarked[item] for item in data.Embarked]

sets = [data, test]

for j in sets:
    for i in range(len(j)):
        """Replacing Sex:male with 1, Sex:female with 0"""
        sex = j.Sex[i]
        if sex == "male":
            j.Sex[i] = 1
        else:
            j.Sex[i] = 0
        """Replacing Cabin:nan with 0, Cabin:str with 1"""
        cabin = j.Cabin[i]
        if type(cabin) == float:
            j.Cabin[i] = 0
        else:
            j.Cabin[i] = 1
        """Replacing Embarked:Q with 0, Embarked:C with 1, Embarked:S with 2"""
        embarked = j.Embarked[i]
        if embarked == "Q":
            j.Embarked[i] = 0
        elif embarked == "C":
            j.Embarked[i] = 1
        else:
            j.Embarked[i] = 2
data = sets[0]
test = sets[1]

data = data.fillna(data.mean())
test = test.fillna(test.mean())

print(data)
print(test)

y = data["Survived"]
pos = y ==1; neg = y == 0

plt.hist(data.Age[pos], color = "red")
plt.hist(data.Age[neg], color = "blue", alpha=0.3)
plt.show()

"""CV Set Creation"""
data_CV = data.sample(frac=0.25)
data_CV.to_csv("./data/CV.csv",  sep=",", index=False)

"""Training Set Creation"""
data = data.drop(data_CV.index)
data.to_csv("./data/train.csv", sep=",", index=False)

"""Test Set Creation"""
test.to_csv("./data/test.csv", sep=",", index=False)