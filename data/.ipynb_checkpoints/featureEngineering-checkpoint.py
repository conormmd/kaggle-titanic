import numpy as np
import pandas as pd

data_train = pd.read_csv('./data/raw_train.csv')
data_test = pd.read_csv('./data/raw_test.csv')

def featureEngineering(data):

    """Dealing with Embarked"""
    Embarked_C = (data.Embarked=="C").astype(int)
    Embarked_S = (data.Embarked=="S").astype(int)
    Embarked_Q = (data.Embarked=="Q").astype(int)

    data = data.drop("Embarked", axis=1)

    """Dealing with Sex"""
    Sex = (data.Sex=="male").astype(int)
    data["Sex"] = Sex

    """Dealing with Cabin"""
    data.Cabin = data.Cabin.str.slice(stop=1) #Slice the first 1 characters, discarding the rest
    data.Cabin = data.Cabin.replace({float(np.nan): 0}) #Turn all nan into 0

    Cabin_A = (data.Cabin=="A").astype(int)
    Cabin_B = (data.Cabin=="B").astype(int)
    Cabin_C = (data.Cabin=="C").astype(int)
    Cabin_D = (data.Cabin=="D").astype(int)
    Cabin_E = (data.Cabin=="E").astype(int)
    Cabin_F = (data.Cabin=="F").astype(int)

    data["Cabin_A"] = Cabin_A
    data["Cabin_B"] = Cabin_B
    data["Cabin_C"] = Cabin_C
    data["Cabin_D"] = Cabin_D
    data["Cabin_E"] = Cabin_E
    data["Cabin_F"] = Cabin_F

    data = data.drop("Cabin", axis=1)

    """Dealing with Names"""
    Names = data.Name.str.split(pat = ",", expand = True) #Split the Names by comma (seperates last & first)
    Names = Names[1].str.split(pat =".", expand = True) #Split the Names by fullstop (seperates title & first)
    data.Name = Names[0].str.lstrip() #Strips the leading character which is a space

    Name_unmarried = (data.Name== ("Miss" or "Master" or "Mme" or "Mml") ).astype(int)
    Name_posh = (data.Name== ("Don" or "Lady" or "Sir" or "the Countess" or "Jonkheer") ).astype(int)
    Name_military = (data.Name== ("Major" or "Col" or "Capt") ).astype(int)
    Name_service = (data.Name== ("Dr", "Rev")).astype(int)

    data["Name_unmarried"] = Name_unmarried
    data["Name_posh"] = Name_posh
    data["Name_military"] = Name_military
    data["Name_service"] = Name_service

    data = data.drop("Name", axis = 1)

    """Dealing with Ticket"""
    data = data.drop("Ticket", axis = 1)

    """Dealing with remaining nan"""
    data = data.fillna(data.mean())

    return data

data = featureEngineering(data_train)
test = featureEngineering(data_test)

"""CV Set"""
data_CV = data.sample(frac=0.25)
data_CV.to_csv("./data/CV.csv",  sep=",", index=False)

"""Train Set"""
data = data.drop(data_CV.index)
data.to_csv("./data/train.csv", sep=",", index=False)

"""Test Set"""
test.to_csv("./data/test.csv", sep=",", index=False)