import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_raw = pd.read_csv('./data/raw_train.csv')
data = data_raw.drop(["PassengerId","Name", "Ticket", "Cabin", "Embarked"], axis=1)

sex = {"male" : 1, "female" : 0}
data.Sex = [sex[item] for item in data.Sex]

data = data.fillna(data.mean())

print(data)

y = data["Survived"]
pos = y ==1; neg = y == 0

plt.hist(data.Age[pos], color = "red")
plt.hist(data.Age[neg], color = "blue", alpha=0.3)
plt.show()

data_CV = data.sample(frac=0.25)
data_CV.to_csv("./data/CV.csv",  sep=",", index=False)

data = data.drop(data_CV.index)
data.to_csv("./data/train.csv", sep=",", index=False)

