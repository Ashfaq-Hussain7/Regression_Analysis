# prompt: also produce the above code using stochastic gradient descent

import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Load the data from the CSV file
data = pd.read_csv('/content/Untitled spreadsheet - Sheet2.csv')
print(data)
m = np.arange(-10, 10, 0.1)
c = np.arange(-10, 10, 0.1)
mc_set = list(product(m, c))
print(mc_set)

errors = []
for mi, ci in mc_set:

    error = np.sum(data['z'] - (mi * data['x'] + ci))

    errors.append(error)

print(errors)

# Load the data
df = pd.read_csv('/content/Untitled spreadsheet - Sheet2.csv')

# Define ranges for m and c
m_values = np.arange(-10, 10, 0.1)
c_values = np.arange(-10, 10, 0.1)

# Extract x and y values from the DataFrame
x_values = df.iloc[:, 0].values
y_values = df.iloc[:, 1].values

# Create a mesh grid for m and c values
M, C = np.meshgrid(m_values, c_values)
errors = np.zeros(M.shape)

# Calculate the errors
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        mi = M[i, j]
        ci = C[i, j]
        predicted_y = mi * x_values + ci
        errors[i, j] = np.sum(np.abs(y_values - predicted_y))  # Sum of absolute errors

# Create the 3D plot using Plotly
fig = go.Figure(data=[go.Surface(z=errors, x=M, y=C)])

# Set labels and title
fig.update_layout(
    scene=dict(
        xaxis_title='m (slope)',
        yaxis_title='c (intercept)',
        zaxis_title='Error'
    ),
    title='Error Surface Plot'
)

# Show the plot
fig.show()


# Load the data
df = pd.read_csv('/content/Untitled spreadsheet - Sheet2.csv')

# Define ranges for m and c
m_values = np.arange(-10, 10, 0.1)
c_values = np.arange(-10, 10, 0.1)

# Extract x and y values from the DataFrame
x_values = df.iloc[:, 0].values
y_values = df.iloc[:, 1].values

# Create a mesh grid for m and c values
M, C = np.meshgrid(m_values, c_values)
errors = np.zeros(M.shape)

# Calculate the errors for the entire grid
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        mi = M[i, j]
        ci = C[i, j]
        predicted_y = mi * x_values + ci
        errors[i, j] = np.sum(np.abs(y_values - predicted_y)) / len(y_values)  # Mean Absolute Error (MAE)

# Function to compute MAE and its gradients
def compute_error_and_gradients(m, c, x_values, y_values):
    N = len(y_values)
    predicted_y = m * x_values + c
    error = np.sum(np.abs(y_values - predicted_y)) / N  # Mean Absolute Error
    error_gradient_m = -np.sum(np.sign(y_values - predicted_y) * x_values) / N  # Gradient w.r.t. m
    error_gradient_c = -np.sum(np.sign(y_values - predicted_y)) / N  # Gradient w.r.t. c
    return error, error_gradient_m, error_gradient_c

# Stochastic Gradient Descent parameters
learning_rate = 0.01
num_iterations = 1000
m_current = 0  # Starting value for m (slope)
c_current = 0  # Starting value for c (intercept)
batch_size = 1  # Mini-batch size (for stochastic, use 1)

# To store the path of m and c values
m_history = [m_current]
c_history = [c_current]
error_history = []

# Stochastic Gradient Descent loop
for i in range(num_iterations):
    random_indices = np.random.choice(len(x_values), size=batch_size, replace=False)
    x_batch = x_values[random_indices]
    y_batch = y_values[random_indices]

    error, grad_m, grad_c = compute_error_and_gradients(m_current, c_current, x_batch, y_batch)
    m_current -= learning_rate * grad_m  # Update m
    c_current -= learning_rate * grad_c  # Update c

    # Save the values at each iteration for plotting
    m_history.append(m_current)
    c_history.append(c_current)
    error_history.append(error)

# Create the 3D plot of the error surface
fig = go.Figure(data=[go.Surface(z=errors, x=M, y=C, colorscale='Viridis', opacity=0.7)])

# Plot the path of stochastic gradient descent on the surface
error_path = np.zeros(len(m_history))
for i in range(len(m_history)):
    predicted_y = m_history[i] * x_values + c_history[i]
    error_path[i] = np.sum(np.abs(y_values - predicted_y)) / len(y_values)  # Error at each step

# Add the gradient descent path
fig.add_trace(go.Scatter3d(
    x=m_history,
    y=c_history,
    z=error_path,
    mode='lines+markers',
    line=dict(color='red', width=4),
    marker=dict(size=5)
))

# Set layout and titles
fig.update_layout(
    scene=dict(
        xaxis_title='m (slope)',
        yaxis_title='c (intercept)',
        zaxis_title='Error'
    ),
    title='Error Surface and Stochastic Gradient Descent Path'
)

# Show the 3D plot
fig.show()

# Plot error over iterations
plt.figure()
plt.plot(error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over Iterations during Stochastic Gradient Descent')
plt.show()
