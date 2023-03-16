import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# What: consist of 2 seperate sets, Training set train ml model on existing observations
# Test set for evaluate of the performance of the model on new observations
from sklearn.model_selection import train_test_split
# x_train number of years of experience, y_test contains the real salaries
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
# Regression have to predict a contineous real value
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
# y_pred predicted salaries
y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
# No need to replace this as the regression line will remain the same
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()