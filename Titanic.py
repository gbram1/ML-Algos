import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

raw_training_set = open('train.csv', 'rt')
raw_test_set = open('test.csv', 'rt')
training_set = pd.read_csv(raw_training_set)
test_set = pd.read_csv(raw_test_set)

X_train, y_train = training_set.drop(['Survived'], axis = 1), training_set['Survived']

dataset = X_train.copy()
dataset['Age'].value_counts()
dataset['Age'].plot(kind = "bar", alpha = 0.1)
dataset = dataset.drop(['Name'], axis = 1)



#group ages into categories
for x in dataset['Age']:
    if x < 20:
        x = 0
    elif x >= 20 and x < 40:
        x = 1
    elif x >= 40 and x < 65:
        x = 2
    else: x =3  
    
    






















