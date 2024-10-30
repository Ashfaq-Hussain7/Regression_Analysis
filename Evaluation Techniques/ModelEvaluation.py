import numpy as np

# Sample dataset (Replace this with your own data)
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 3.5 * X[:, 0] + 2.3 * X[:, 1] + 5.1 * X[:, 2] + np.random.randn(100)  # True model with some noise


class LinearRegressionScratch:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Adding intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculating coefficients using the Normal Equation
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Adding intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coefficients

# Train the model
model = LinearRegressionScratch()
model.fit(X, y)
y_pred = model.predict(X)


def r_squared(y, y_pred):
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    return 1 - (ss_residual / ss_total)

r2 = r_squared(y, y_pred)
print("R^2:", r2)


def mallows_cp(y, y_pred, p):
    n = len(y)
    sse = np.sum((y - y_pred) ** 2)
    sigma_squared = sse / (n - p - 1)
    return sse / sigma_squared - (n - 2 * p)

p = X.shape[1]  # Number of features
cp = mallows_cp(y, y_pred, p)
print("Mallows' Cp:", cp)


def aic(y, y_pred, p):
    n = len(y)
    sse = np.sum((y - y_pred) ** 2)
    return n * np.log(sse / n) + 2 * (p + 1)

aic_value = aic(y, y_pred, p)
print("AIC:", aic_value)


def bic(y, y_pred, p):
    n = len(y)
    sse = np.sum((y - y_pred) ** 2)
    return n * np.log(sse / n) + (p + 1) * np.log(n)

bic_value = bic(y, y_pred, p)
print("BIC:", bic_value)
