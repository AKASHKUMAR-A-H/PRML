import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
n = 10000
training_ratio = 0.80

# Three class means
MU = np.array([
    [-2.0, -1.0],
    [ 4.0,  0.0],
    [ 1.0,  5.0]
])

# Gaussian
def normal_values(size):
    u1 = np.random.random(size)
    u2 = np.random.random(size)
    u1 = np.maximum(u1, 1e-12)
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


def make_gaussian_samples(mean, covariance, count):
    z = np.column_stack([
        normal_values(count),
        normal_values(count)
    ])

    # Eigen-decomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    transform = (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    return z @ transform.T + mean

# Data Generation
def create_dataset(covariance_list):
    data = []
    labels = []
    for class_id in range(3):
        points = make_gaussian_samples(
            MU[class_id],
            covariance_list[class_id],
            n
        )
        data.append(points)
        labels.append(
            np.full(
                n,
                class_id
            )
        )
    return np.vstack(data), np.concatenate(labels)

# 80% Training and 20% Testing
def train_test_split_manual(X, y):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for class_id in range(3):
        class_data = X[y == class_id]
        order = np.random.permutation(len(class_data))
        class_data = class_data[order]
        cut = int(training_ratio * len(class_data))

        train_x.append(class_data[:cut])
        test_x.append(class_data[cut:])

        train_y.append(np.full(cut, class_id))

        test_y.append(np.full(len(class_data) - cut, class_id))

    return (
        np.vstack(train_x),
        np.concatenate(train_y),
        np.vstack(test_x),
        np.concatenate(test_y)
    )

# Maximum Likelihood Estimate Mean
def mle_mean(X):
    return np.sum(X, axis=0) / len(X)

# Maximum Likelihood Estimate Covariance
def mle_covariance(X, mean):
    centered = X - mean
    return (centered.T @ centered) / len(X)

# Parameter estimation
def estimate_class_parameters(X, y):
    class_means = []
    class_covariances = []
    for class_id in range(3):
        class_points = X[y == class_id]
        mean = mle_mean(class_points)
        covariance = mle_covariance(
            class_points,
            mean
        )
        class_means.append(mean)
        class_covariances.append(covariance)
    return class_means, class_covariances

# Constrain Covariance
def force_isotropic(covariance):
    variance = np.trace(covariance) / 2
    return variance * np.eye(2)

def force_diagonal(covariance):
    return np.diag(np.diag(covariance))

# Creating Required Model
def build_model(X_train, y_train,covariance_case):
    means, individual_covariances = (
        estimate_class_parameters(
            X_train,
            y_train
        )
    )

    # Shared covariance
    if covariance_case == "shared_isotropic":
        total = np.zeros((2, 2))
        number_of_points = 0
        for class_id in range(3):
            class_points = X_train[y_train == class_id]
            centered = (class_points - means[class_id])
            total += centered.T @ centered
            number_of_points += len(class_points)

        common_covariance = (total / number_of_points)
        common_covariance = force_isotropic(common_covariance)
        covariances = [
            common_covariance.copy()
            for _ in range(3)
        ]

    # Different isotropic covariance
    elif covariance_case == "different_isotropic":
        covariances = [
            force_isotropic(c)
            for c in individual_covariances
        ]

    # Shared diagonal covariance
    elif covariance_case == "shared_diagonal":
        total = np.zeros((2, 2))
        number_of_points = 0
        for class_id in range(3):
            class_points = X_train[
                y_train == class_id
            ]

            centered = (
                class_points
                - means[class_id]
            )

            total += centered.T @ centered
            number_of_points += len(class_points)
        common_covariance = (
            total / number_of_points
        )
        common_covariance = force_diagonal(
            common_covariance
        )
        covariances = [
            common_covariance.copy()
            for _ in range(3)
        ]

    # Different diagonal covariance
    elif covariance_case == "different_diagonal":
        covariances = [
            force_diagonal(c)
            for c in individual_covariances
        ]

    # Shared full covariance
    elif covariance_case == "shared_full":
        total = np.zeros((2, 2))
        number_of_points = 0
        for class_id in range(3):
            class_points = X_train[
                y_train == class_id
            ]
            centered = (
                class_points
                - means[class_id]
            )
            total += centered.T @ centered
            number_of_points += len(class_points)
        common_covariance = (
            total / number_of_points
        )
        covariances = [
            common_covariance.copy()
            for _ in range(3)
        ]

    # Different full covariance
    elif covariance_case == "different_full":
        covariances = individual_covariances

    else:
        raise ValueError(
            "Invalid covariance case"
        )
    return means, covariances

# Log Gaussian Likelihood
def log_probability(X, mean, covariance):
    covariance = (
        covariance
        + 1e-10 * np.eye(2)
    )
    difference = X - mean
    inverse = np.linalg.inv(covariance)
    determinant = np.linalg.det(covariance)
    mahalanobis = np.sum(
        (difference @ inverse) * difference,
        axis=1
    )
    return -0.5 * (
        2 * np.log(2 * np.pi)
        + np.log(determinant)
        + mahalanobis
    )

# Classification
def predict(X, means, covariances):
    scores = []
    for mean, covariance in zip(
        means,
        covariances
    ):
        scores.append(
            log_probability(
                X,
                mean,
                covariance
            )
        )

    scores = np.array(scores)
    return np.argmax(
        scores,
        axis=0
    )

# Confusion Matrix
def get_confusion_matrix(actual, predicted):
    matrix = np.zeros(
        (3, 3),
        dtype=int
    )
    for a, p in zip(actual, predicted):
        matrix[a, p] += 1
    return matrix

# Accuracy
def calculate_accuracy(matrix):
    return (np.trace(matrix) / np.sum(matrix))

# Decision Region

def draw_boundary(
    X_test,
    y_test,
    means,
    covariances,
    title
):

    margin = 1.5

    x1_min = X_test[:, 0].min() - margin
    x1_max = X_test[:, 0].max() + margin

    x2_min = X_test[:, 1].min() - margin
    x2_max = X_test[:, 1].max() + margin

    x1 = np.linspace(
        x1_min,
        x1_max,
        350
    )

    x2 = np.linspace(
        x2_min,
        x2_max,
        350
    )

    xx, yy = np.meshgrid(x1, x2)

    grid = np.column_stack([
        xx.ravel(),
        yy.ravel()
    ])

    prediction = predict(
        grid,
        means,
        covariances
    )

    prediction = prediction.reshape(
        xx.shape
    )

    plt.figure(figsize=(9, 7))

    plt.contourf(
        xx,
        yy,
        prediction,
        levels=np.arange(-0.5, 3.5, 1),
        alpha=0.20,
        cmap="viridis"
    )

    symbols = ["o", "s", "^"]

    for class_id in range(3):

        points = X_test[
            y_test == class_id
        ]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=5,
            marker=symbols[class_id],
            alpha=0.35,
            label=f"Class {class_id + 1}"
        )

    for class_id, mean in enumerate(means):

        plt.scatter(
            mean[0],
            mean[1],
            marker="X",
            s=180,
            edgecolors="black",
            linewidths=2
        )

        plt.annotate(
            f"μ{class_id + 1}",
            mean,
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=11
        )

    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

# One Complete Experiment

def execute_experiment(
    name,
    true_covariances,
    covariance_case
):
    print("\n")
    print("=" * 75)
    print(name)
    print("=" * 75)

    # Generate 30,000 samples
    X, y = create_dataset(true_covariances)

    print(
        "Total samples generated:",
        len(X)
    )

    # Split
    X_train, y_train, X_test, y_test = (
        train_test_split_manual(X,y)
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

    # Estimate parameters
    estimated_means, estimated_covariances = (
        build_model(
            X_train,
            y_train,
            covariance_case
        )
    )

    # Display parameters
    print("\nEstimated Means")

    for i, mean in enumerate(
        estimated_means
    ):

        print(
            f"Class {i + 1}:",
            np.round(mean, 4)
        )

    print("\nEstimated Covariance Matrices")

    for i, covariance in enumerate(
        estimated_covariances
    ):

        print(
            f"\nClass {i + 1}:"
        )

        print(
            np.round(covariance, 4)
        )

    # Classification
    predicted = predict(
        X_test,
        estimated_means,
        estimated_covariances
    )

    # Confusion matrix
    matrix = get_confusion_matrix(
        y_test,
        predicted
    )

    accuracy = calculate_accuracy(
        matrix
    )

    print("\nConfusion Matrix")
    print(matrix)

    print(
        f"\nAccuracy = {accuracy * 100:.2f}%"
    )

    # Plot
    draw_boundary(
        X_test,
        y_test,
        estimated_means,
        estimated_covariances,
        name
    )
    return accuracy

# Covariance Matrix

# Shared Isotropic
COV_SHARED_ISO = [
    np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
] * 3


# Different Isotropic

COV_DIFFERENT_ISO = [
    np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),
    np.array([
        [2.0, 0.0],
        [0.0, 2.0]
    ]),
    np.array([
        [0.5, 0.0],
        [0.0, 0.5]
    ])
]

# Shared Diagonal
COV_SHARED_DIAGONAL = [
    np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ])
] * 3

# Different Diagonal
COV_DIFFERENT_DIAGONAL = [
    np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ]),
    np.array([
        [2.0, 0.0],
        [0.0, 1.0]
    ]),
    np.array([
        [1.5, 0.0],
        [0.0, 0.5]
    ])
]

# Shared Full
COV_SHARED_FULL = [
    np.array([
        [2.0, -0.8],
        [-0.8, 1.5]
    ])
] * 3

# Different Full
COV_DIFFERENT_FULL = [
    np.array([
        [2.0, -0.8],
        [-0.8, 1.5]
    ]),
    np.array([
        [1.5, 0.5],
        [0.5, 2.0]
    ]),
    np.array([
        [2.0, 0.7],
        [0.7, 1.0]
    ])
]

# Running all cases

results = {}

results["Shared Isotropic"] = execute_experiment(
    "Case 1 - Shared Isotropic Covariance",
    COV_SHARED_ISO,
    "shared_isotropic"
)

results["Different Isotropic"] = execute_experiment(
    "Case 2 - Different Isotropic Covariance",
    COV_DIFFERENT_ISO,
    "different_isotropic"
)

results["Shared Diagonal"] = execute_experiment(
    "Case 3 - Shared Diagonal Covariance",
    COV_SHARED_DIAGONAL,
    "shared_diagonal"
)

results["Different Diagonal"] = execute_experiment(
    "Case 4 - Different Diagonal Covariance",
    COV_DIFFERENT_DIAGONAL,
    "different_diagonal"
)

results["Shared Full"] = execute_experiment(
    "Case 5 - Shared Full Covariance",
    COV_SHARED_FULL,
    "shared_full"
)

results["Different Full"] = execute_experiment(
    "Case 6 - Different Full Covariance",
    COV_DIFFERENT_FULL,
    "different_full"
)

# Summary
print("\n")
print("=" * 75)
print("FINAL ACCURACY SUMMARY")
print("=" * 75)
for case, accuracy in results.items():
    print(
        f"{case:<30} : "
        f"{accuracy * 100:.2f}%"
    )
