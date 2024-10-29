import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# Batch Gradient Descent parameters
learning_rate = 0.01
num_iterations = 1000
m_current = 0  # Starting value for m (slope)
c_current = 0  # Starting value for c (intercept)
batch_size = len(x_values)  # Batch size for Batch Gradient Descent (use full dataset)

# To store the path of m and c values
m_history = [m_current]
c_history = [c_current]
error_history = []

# Batch Gradient Descent loop
for i in range(num_iterations):
    error, grad_m, grad_c = compute_error_and_gradients(m_current, c_current, x_values, y_values)
    m_current -= learning_rate * grad_m  # Update m
    c_current -= learning_rate * grad_c  # Update c

    # Save the values at each iteration for plotting
    m_history.append(m_current)
    c_history.append(c_current)
    error_history.append(error)

# Create the 3D plot of the error surface
fig = go.Figure(data=[go.Surface(z=errors, x=M, y=C, colorscale='Viridis', opacity=0.7)])

# Plot the path of batch gradient descent on the surface
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
    line=dict(color='blue', width=4),
    marker=dict(size=5)
))

# Set layout and titles
fig.update_layout(
    scene=dict(
        xaxis_title='m (slope)',
        yaxis_title='c (intercept)',
        zaxis_title='Error'
    ),
    title='Error Surface and Batch Gradient Descent Path'
)

# Show the 3D plot
fig.show()

# Plot error over iterations
plt.figure()
plt.plot(error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over Iterations during Batch Gradient Descent')
plt.show()