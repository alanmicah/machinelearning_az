import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the libraries
dataset = pd.read_csv('Data.csv')
# iloc locates indexes
# [:] always includes the lower bound (before the colon)
# and exlcudes the upper bound (after the colon)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# What: consist of 2 seperate sets, Training set train ml model on existing observations
# Test set for evaluate of the performance of the model on new observations
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)