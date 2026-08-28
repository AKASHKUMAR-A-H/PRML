import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("noisy_12.txt")
x = data[:, 0]
y = data[:, 1]

# Shuffle the data
np.random.seed(42)
indices = np.random.permutation(len(x))
x = x[indices]
y = y[indices]

# Splitting the data(60% Training, 20% Testing, 20% Validation)
n = len(x)
train_end = int(0.60 * n)
test_end = int(0.80 * n)

x_train = x[:train_end]
y_train = y[:train_end]

x_test = x[train_end:test_end]
y_test = y[train_end:test_end]

x_val = x[test_end:]
y_val = y[test_end:]

print("Training samples   :", len(x_train))
print("Testing samples    :", len(x_test))
print("Validation samples :", len(x_val))

# Creating polynomial design matrix
def matrix(x, degree):
    X = np.ones((len(x), degree + 1))
    for i in range(1, degree + 1):
        X[:, i] = x ** i
    return X

# Normal Equation
def normal_equation(X, y):
    XT = X.T
    XTX = XT @ X
    XTy = XT @ y

    w = np.linalg.inv(XTX) @ XTy

    return w

def predict(X, w):        # Prediction
    return X @ w

def mse(y_actual, y_predicted):                       # Mean Squared Error(MSE)
    return np.mean((y_actual - y_predicted) ** 2)

# Train model for different degrees
degrees = range(1, 11)
test_errors = []
weights = []
best_degree = None
best_test_error = float("inf")
best_weight = None


for degree in degrees:
    
    X_train = matrix(x_train, degree)    # Create training matrix
    w = normal_equation(X_train, y_train)     # Find weights using Normal Equation
    X_test = matrix(x_test, degree)      # Create test matrix
    y_test_pred = predict(X_test, w)      # Predict test values

    test_error = mse(y_test, y_test_pred)     # Calculate test MSE
    test_errors.append(test_error)
    weights.append(w)
    print("Degree:", degree,
          " Test MSE:", round(test_error, 4))

    # Find best degree
    if test_error < best_test_error:
        best_test_error = test_error
        best_degree = degree
        best_weight = w

# Validation using best degree
X_val = matrix(x_val, best_degree)
y_val_pred = predict(X_val, best_weight)
validation_error = mse(y_val, y_val_pred)

print("Best polynomial degree :", best_degree)
print("Best test MSE          :", best_test_error)
print("Validation MSE         :", validation_error)
print("\nWeights:")

for i in range(len(best_weight)):
    print("w", i, "=", best_weight[i])

# Plotting Test MSE vs Polynomial Degree
plt.figure(figsize=(9, 5))
plt.plot(
    list(degrees),
    test_errors,
    marker='o'
)
plt.scatter(
    best_degree,
    best_test_error,
    s=100
)
plt.xlabel("Polynomial Degree")
plt.ylabel("Test MSE")
plt.title("Effect of Polynomial Degree on Test Error")
plt.xticks(list(degrees))
plt.grid(True)
plt.tight_layout()
plt.show()

# Create X values for curve
x_plot = np.linspace(
    np.min(x),
    np.max(x),
    1000
)

# Plotting Fitted curves for differnt degrees
selected_degrees = [1, 3, 5, 7, 10]
plt.figure(figsize=(12, 7))

# Plot original noisy data
plt.scatter(
    x,
    y,
    s=8,
    alpha=0.25,
    label="Noisy data"
)

# Plot polynomial curves
for degree in selected_degrees:
    
    index = degree - 1      # Get corresponding weight
    w = weights[index]
    X_plot = matrix(x_plot, degree)      # Create polynomial matrix
    y_plot = predict(X_plot, w)     # Calculate predictions

    # Plot fitted curve
    plt.plot(
        x_plot,
        y_plot,
        linewidth=2,
        label="Degree " + str(degree)
    )
plt.xlabel("x")
plt.ylabel("y")
plt.title("Effect of Polynomial Degree on Fitted Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best model plot
X_plot = matrix(x_plot, best_degree)
y_plot = predict(X_plot, best_weight)
plt.figure(figsize=(10, 6))
plt.scatter(
    x_train,
    y_train,
    s=8,
    alpha=0.25,
    label="Training data"
)

plt.plot(
    x_plot,
    y_plot,
    linewidth=3,
    label="Best polynomial (degree "
          + str(best_degree) + ")"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title(
    "Best Polynomial Regression Fit"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
