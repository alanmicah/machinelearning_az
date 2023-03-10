# Data Preprocessing

# Import the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
# ifelse First parameter is the if condition
# second parameter value input if value is true
# third parameter value input if value is false

# Checking all values in the column Age
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
