# Assignment_6-Scikit-Learn-Regression

## Purpose
The purpose of this project is to build and evaluate different machine learning regression models using the built=in dataset from Scikit Learn.

## Models Used
Three regression models were trained and tested:

- Linear Regression
- Decision Tree Regressor
- K-Neighbors Regressor

## Process
The diabetes dataset was loaded from Scikit Learn and split into training and testing sets using train_test_split. Each model was trained using the training data and then used to make predictions on the test data.

The models were evaluated using three common metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Results
Linear Regression performed the best overall. It had the lowest MAE and MSE values and the highest R² score. This indicates that it predicted the target values more accurately than the other models.

The Decision Tree model performed the worst and produced a negative R² value, which means it performed worse than predicting the average value.

K-Neighbors Regressor performed better than the Decision Tree but still had higher error values compared to Linear Regression.
  
