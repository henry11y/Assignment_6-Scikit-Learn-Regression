from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load dataset
data = load_diabetes()
X = data.data
y = data.target

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create models
linear = LinearRegression()
tree = DecisionTreeRegressor()
knn = KNeighborsRegressor()

# train models
linear.fit(X_train, y_train)
tree.fit(X_train, y_train)
knn.fit(X_train, y_train)

# predictions
linear_pred = linear.predict(X_test)
tree_pred = tree.predict(X_test)
knn_pred = knn.predict(X_test)

print("Linear Regression Results")
print("MAE:", mean_absolute_error(y_test, linear_pred))
print("MSE:", mean_squared_error(y_test, linear_pred))
print("R2:", r2_score(y_test, linear_pred))
print()

print("Decision Tree Results")
print("MAE:", mean_absolute_error(y_test, tree_pred))
print("MSE:", mean_squared_error(y_test, tree_pred))
print("R2:", r2_score(y_test, tree_pred))
print()

print("KNN Results")
print("MAE:", mean_absolute_error(y_test, knn_pred))
print("MSE:", mean_squared_error(y_test, knn_pred))
print("R2:", r2_score(y_test, knn_pred))
print()