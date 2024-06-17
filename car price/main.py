import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Load the dataset into a DataFrame
car_data = pd.read_csv('data.csv')

# Display the first few rows of the DataFrame
print(car_data.head())

# Check the dimensions of the DataFrame
print(car_data.shape)

# Get detailed information about the DataFrame
print(car_data.info())

# Check for missing values in the DataFrame
print(car_data.isnull().sum())

# Display the distribution of categorical columns
print(car_data['Fuel_Type'].value_counts())
print(car_data['Seller_Type'].value_counts())
print(car_data['Transmission'].value_counts())

# Encode categorical columns
car_data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_data.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_data.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Define feature matrix (X) and target vector (Y)
X = car_data.drop(columns=['Car_Name', 'Selling_Price'], axis=1)
Y = car_data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# Predict on the training set
train_predictions_lr = linear_model.predict(X_train)

# Calculate R squared error for training set
r2_train_lr = metrics.r2_score(Y_train, train_predictions_lr)
print("R squared Error (Training) : ", r2_train_lr)

# Plot actual vs predicted prices for training set
plt.scatter(Y_train, train_predictions_lr)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Training Set) - Linear Regression")
plt.show()

# Predict on the testing set
test_predictions_lr = linear_model.predict(X_test)

# Calculate R squared error for testing set
r2_test_lr = metrics.r2_score(Y_test, test_predictions_lr)
print("R squared Error (Testing) : ", r2_test_lr)

# Plot actual vs predicted prices for testing set
plt.scatter(Y_test, test_predictions_lr)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Testing Set) - Linear Regression")
plt.show()

# Initialize and train the Lasso Regression model
lasso_model = Lasso()
lasso_model.fit(X_train, Y_train)

# Predict on the training set
train_predictions_lasso = lasso_model.predict(X_train)

# Calculate R squared error for training set
r2_train_lasso = metrics.r2_score(Y_train, train_predictions_lasso)
print("R squared Error (Training) : ", r2_train_lasso)

# Plot actual vs predicted prices for training set
plt.scatter(Y_train, train_predictions_lasso)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Training Set) - Lasso Regression")
plt.show()

# Predict on the testing set
test_predictions_lasso = lasso_model.predict(X_test)

# Calculate R squared error for testing set
r2_test_lasso = metrics.r2_score(Y_test, test_predictions_lasso)
print("R squared Error (Testing) : ", r2_test_lasso)

# Plot actual vs predicted prices for testing set
plt.scatter(Y_test, test_predictions_lasso)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Testing Set) - Lasso Regression")
plt.show()
