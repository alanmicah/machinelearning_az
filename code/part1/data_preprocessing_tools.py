# Library an assemble of classes and functions
# Allows us to work with arrays
from cv2 import transform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import im

# Import the libraries
dataset = pd.read_csv('Data.csv')
# iloc locates indexes
# [:] always includes the lower bound (before the colon)
# and exlcudes the upper bound (after the colon)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

# Taking care of the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
# this returns
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Only encode the column at index 0
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# we want to transform x
x = np.array(ct.fit_transform(x))

print(x)

#  Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
# What: consist of 2 seperate sets, Training set train ml model on existing observations
# Test set for evaluate of the performance of the model on new observations
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)
print(x_test)
# 8 purchased descisions. same customers in x_train
print(y_train)
# coresponds to the x_test dataset
print(y_test)

# We apply feacture scaling afterwards a technique to get the mean and standard deviation
# If we did it before then we would grab information (information leakage) from the test set
# which we aren't suppose to know
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Applies feature scaling
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
# Only apply the transform method
# because the features need to be scaled by the same scaler as the training set
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)