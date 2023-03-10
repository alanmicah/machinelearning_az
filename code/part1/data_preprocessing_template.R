# Data Preprocessing

# Import the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

# Taking care of missing data
# ifelse First parameter is the if condition
# second parameter value input if value is true
# third parameter value input if value is false

# # Checking all values in the column Age
# dataset$Age = ifelse(is.na(dataset$Age),
#                      ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Age)
# dataset$Salary = ifelse(is.na(dataset$Salary),
#                      ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Salary)

# # Encoding categorical data
# # c() is a vector
# # factor is not a numeric number
# dataset$Country = factor(dataset$Country,
#                          levels = c('France', 'Spain', 'Germany'),
#                          labels = c(1,2,3))
# dataset$Purchased = factor(dataset$Purchased,
#                          levels = c('No', 'Yes'),
#                          labels = c(0,1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
# Splitting the data to prevent Overfitting
# It will return True observation chosen to go to Training set
# or False observation chosen to go to Test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# # Euclidean Distance will be dominated by the Salary (so much larger)
# # Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

