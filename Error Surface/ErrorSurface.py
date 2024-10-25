import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
