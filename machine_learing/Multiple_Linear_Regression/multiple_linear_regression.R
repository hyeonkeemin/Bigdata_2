# Multiple Linear Regression

dataset = read.csv('50_Startups.csv')

# categorical data 인코딩
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

#  Training set, Test set 스플릿
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# 피쳐스케일링
training_set = scale(training_set)
test_set = scale(test_set)

# Multiple Linear Regression에 Training set 피팅
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Test set 결과에 Predicting
y_pred = predict(regressor, newdata = test_set)