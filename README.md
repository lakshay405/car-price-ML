# car-price-ML
Car Price Prediction Models
This project focuses on predicting the selling price of used cars using two different regression models: Linear Regression and Lasso Regression. The goal is to build models that accurately estimate the selling price based on various features such as car specifications, fuel type, seller type, and transmission type.

Dataset
The dataset (data.csv) includes information about used cars such as their make, model, mileage, fuel type, seller type, transmission type, and selling price.

Workflow
Data Loading and Preprocessing:

Load the dataset into a Pandas DataFrame (car_data).
Check the dimensions and structure of the dataset using shape and info.
Handle missing values and encode categorical variables (Fuel_Type, Seller_Type, Transmission) to numerical values suitable for modeling.
Exploratory Data Analysis (EDA):

Explore the distribution of categorical columns (Fuel_Type, Seller_Type, Transmission) to understand the composition of the dataset.
Model Training and Evaluation:

Linear Regression:

Prepare feature matrix (X) and target vector (Y).
Split the dataset into training and testing sets using train_test_split.
Train the Linear Regression model and evaluate its performance using R squared error on both training and testing sets.
Visualize actual vs predicted prices for both training and testing sets.
Lasso Regression:

Repeat the above steps for Lasso Regression, which includes feature selection via regularization.
Train the Lasso Regression model and evaluate its performance using R squared error.
Visualize actual vs predicted prices for both training and testing sets.
Libraries Used
pandas for data manipulation and analysis.
matplotlib and seaborn for data visualization.
sklearn for model selection (train_test_split, LinearRegression, Lasso) and evaluation (metrics).
Conclusion
This project demonstrates the application of two regression techniques, Linear Regression and Lasso Regression, to predict the selling price of used cars based on various features. The models provide insights into how different features impact the pricing and offer a predictive tool for estimating the market value of used cars.

