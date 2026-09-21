import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eig
from scipy.linalg import sqrtm

np.random.seed(42)
N = 1000

# Helper functions

def generate_with_cov(mu, cov, n=N):
    X = np.random.randn(n, 2)
    L = sqrtm(cov)
    return mu + X @ L.T

def estimate_params(data):
    mu_hat = np.mean(data, axis=0)
    cov_hat = np.cov(data, rowvar=False)
    return mu_hat, cov_hat

def plot_constant_density(ax, mu, cov, color='blue'):
    x = np.linspace(mu[0]-6, mu[0]+6, 300)
    y = np.linspace(mu[1]-6, mu[1]+6, 300)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = multivariate_normal(mu, cov).pdf(pos)
    ax.contour(X, Y, Z, levels=7, colors=color, alpha=0.7, linewidths=1.2)

def plot_eigenvectors(ax, mu, cov, scale=2.0, color='red'):
    eigenvalues, eigenvectors = eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    for i in range(2):
        length = scale * np.sqrt(eigenvalues[i])
        vec = eigenvectors[:, i] * length
        ax.arrow(mu[0], mu[1],  vec[0],  vec[1],
                 head_width=0.15, head_length=0.2,
                 fc=color, ec=color, linewidth=2.5, length_includes_head=True)
        ax.arrow(mu[0], mu[1], -vec[0], -vec[1],
                 head_width=0.15, head_length=0.2,
                 fc=color, ec=color, linewidth=2.5, length_includes_head=True)

# PART 1: Isotropic Covariance (C = σ²I)
print("\n PART 1: Isotropic Covariance ")

mu = np.array([0.0, 0.0])
cov = np.eye(2)

D1 = np.random.normal(0, 1, N)
D2 = np.random.normal(0, 1, N)
data = np.column_stack((D1, D2))

mu_hat, cov_hat = estimate_params(data)
eigvals, eigvecs = eig(cov_hat)
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]

print("Estimated mean:", mu_hat)
print("Estimated covariance:\n", cov_hat)
print("Eigenvalues:", eigvals)
print("Axis lengths (√λ):", np.sqrt(eigvals))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(data[:,0], data[:,1], s=8, alpha=0.4, c='steelblue', label='Data')
plot_constant_density(ax, mu_hat, cov_hat, 'darkblue')
plot_eigenvectors(ax, mu_hat, cov_hat)
ax.set_title("Part 1: Isotropic Covariance (C = σ²I)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# PART 2: Diagonal Covariance
print("\n PART 2: Diagonal Covariance ")

mu = np.array([1.0, 2.0])
cov = np.array([[3.0, 0.0],
                [0.0, 1.0]])

data = generate_with_cov(mu, cov)
mu_hat, cov_hat = estimate_params(data)
eigvals, eigvecs = eig(cov_hat)
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]

print("Estimated mean:", mu_hat)
print("Estimated covariance:\n", cov_hat)
print("Eigenvalues:", eigvals)
print("Axis lengths (√λ):", np.sqrt(eigvals))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(data[:,0], data[:,1], s=8, alpha=0.4, c='seagreen', label='Data')
plot_constant_density(ax, mu_hat, cov_hat, 'darkgreen')
plot_eigenvectors(ax, mu_hat, cov_hat)
ax.set_title("Part 2: Diagonal Covariance")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# PART 3: Full Covariance
print("\n PART 3: Full Covariance ")

mu = np.array([0.0, 0.0])
cov = np.array([[2.0, 1.2],
                [1.2, 1.5]])

data = generate_with_cov(mu, cov)
mu_hat, cov_hat = estimate_params(data)
eigvals, eigvecs = eig(cov_hat)
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]

print("Estimated mean:", mu_hat)
print("Estimated covariance:\n", cov_hat)
print("Eigenvalues:", eigvals)
print("Axis lengths (√λ):", np.sqrt(eigvals))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(data[:,0], data[:,1], s=8, alpha=0.4, c='darkorange', label='Data')
plot_constant_density(ax, mu_hat, cov_hat, 'chocolate')
plot_eigenvectors(ax, mu_hat, cov_hat)
ax.set_title("Part 3: Full Covariance")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


# PART 4: Decision Boundaries (2 Experiments)
print("\n PART 4: Decision Boundaries ")

#  Experiment 1: Different covariances 
print("\n Experiment 1: Different covariances ")

mu1 = np.array([-2.0, -1.0])
cov1 = np.array([[1.5, 0.6],
                 [0.6, 1.0]])

mu2 = np.array([2.5, 1.5])
cov2 = np.array([[1.0, -0.4],
                 [-0.4, 2.0]])

Y1 = generate_with_cov(mu1, cov1)
Y2 = generate_with_cov(mu2, cov2)

m1, c1 = estimate_params(Y1)
m2, c2 = estimate_params(Y2)

print("Class 1 mean:", m1)
print("Class 2 mean:", m2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Y1[:,0], Y1[:,1], s=10, alpha=0.4, c='royalblue', label='Class ω1')
ax.scatter(Y2[:,0], Y2[:,1], s=10, alpha=0.4, c='crimson', label='Class ω2')
plot_constant_density(ax, m1, c1, 'blue')
plot_constant_density(ax, m2, c2, 'red')

x = np.linspace(-7, 8, 400)
y = np.linspace(-6, 7, 400)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
pdf1 = multivariate_normal(m1, c1).pdf(pos)
pdf2 = multivariate_normal(m2, c2).pdf(pos)
ax.contour(X, Y, pdf1 - pdf2, levels=[0], colors='black', linewidths=2.5, linestyles='--')

ax.set_title("Part 4 - Experiment 1: Different Covariances\n(Quadratic Decision Boundary)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Experiment 2: Same covariance 
print("\n Experiment 2: Same covariance ")

mu1 = np.array([-2.0, 0.0])
mu2 = np.array([2.5, 1.0])
same_cov = np.array([[1.5, 0.4],
                     [0.4, 1.2]])

Y1 = generate_with_cov(mu1, same_cov)
Y2 = generate_with_cov(mu2, same_cov)

m1, c1 = estimate_params(Y1)
m2, c2 = estimate_params(Y2)

print("Both classes use the same covariance")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Y1[:,0], Y1[:,1], s=10, alpha=0.4, c='royalblue', label='Class ω1')
ax.scatter(Y2[:,0], Y2[:,1], s=10, alpha=0.4, c='crimson', label='Class ω2')
plot_constant_density(ax, m1, c1, 'blue')
plot_constant_density(ax, m2, c2, 'red')

x = np.linspace(-7, 8, 400)
y = np.linspace(-5, 6, 400)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
pdf1 = multivariate_normal(m1, c1).pdf(pos)
pdf2 = multivariate_normal(m2, c2).pdf(pos)
ax.contour(X, Y, pdf1 - pdf2, levels=[0], colors='black', linewidths=2.5, linestyles='--')

ax.set_title("Part 4 - Experiment 2: Same Covariance\n(Linear Decision Boundary)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

print("\n CONCLUSION ")
print("Different covariances - Quadratic (curved) boundary")
print("Same covariance       - Linear (straight) boundary")
