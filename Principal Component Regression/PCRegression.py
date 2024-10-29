#Implement Principle Component Regression

import pandas as pd
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Load the CSV dataset (update the file path as needed)
file_path = '/content/predictive_maintenance.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Select relevant features and the target variable
features = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                 'Torque [Nm]', 'Tool wear [min]']].values
target = data['Target'].values  # Assuming 'Target' is the column name for the dependent variable

# Standardize the data (subtract mean and divide by standard deviation)
X_standardized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)


# Step 1: Perform PCA using SVD
U, S, Vt = svd(X_standardized, full_matrices=False)

# Principal components (PCs) are given by U @ S
PCs = U @ np.diag(S)


# Step 2: Choose the number of principal components to retain (e.g., k = 3)
k = 3  # You can adjust k based on explained variance or trial/error
X_pca = PCs[:, :k]  # Use the first k principal components

# Add intercept term to the principal components
X_pca = np.hstack([np.ones((X_pca.shape[0], 1)), X_pca])


# Step 3: Define cost function for linear regression on principal components
def pcr_cost_function(theta, X_pca, y):
    m = len(y)
    predictions = X_pca @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# Step 4: Minimize the cost function to find optimal theta (weights)
initial_theta = np.zeros(X_pca.shape[1])

# Optimize the cost function using scipy's minimize
result = minimize(pcr_cost_function, initial_theta, args=(X_pca, target), method='TNC')

# Optimal parameters (theta) found
optimal_theta = result.x


# Step 5: Predict using the principal components
def predict_pcr(X_pca, theta):
    return X_pca @ theta

# Make predictions on the principal components
predictions = predict_pcr(X_pca, optimal_theta)

# Evaluate model (e.g., R-squared)
rss = np.sum((target - predictions) ** 2)  # Residual sum of squares
tss = np.sum((target - np.mean(target)) ** 2)  # Total sum of squares
r_squared = 1 - rss / tss


# Print the results
print("Optimal theta (weights) for PCR:", optimal_theta)
print(f"R-squared for the PCR model: {r_squared:.4f}")


#PLOTS

# 1. Explained Variance (Scree Plot)
explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(S)+1), explained_variance_ratio, 'o-', linewidth=2)
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.show()


# 2. Residual Plot
residuals = target - predictions
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# 3. Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(target, predictions, alpha=0.7)
plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

