import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== SETTINGS ======================
SQUARE_IMAGE   = "cat_square.png"
RECT_IMAGE     = "cat_rectangle.png"
K_VALUES       = [10, 50, 100]
OUTPUT_DIR     = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ======================================================

def load_grayscale(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64)

def frobenius(A, Ak):
    return np.linalg.norm(A - Ak, 'fro')

def save_triple(original, reconstructed, error, title, filename):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(reconstructed, 0, 255), cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(error, cmap='gray')
    plt.title("|A - Ak|")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def evd_reconstruct(A, k):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort by absolute value (largest first)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # --- Proper handling of complex conjugate pairs ---
    selected = []
    i = 0
    n = len(eigenvalues)
    
    while len(selected) < k and i < n:
        val = eigenvalues[i]
        
        if np.isclose(val.imag, 0, atol=1e-8):          # Real eigenvalue
            selected.append(i)
            i += 1
        else:
            # Complex eigenvalue → take the conjugate pair together
            if len(selected) + 2 <= k:
                selected.append(i)
                # The conjugate is usually the next one after magnitude sorting
                if i + 1 < n and np.isclose(eigenvalues[i+1], np.conj(val), atol=1e-6):
                    selected.append(i + 1)
                    i += 2
                else:
                    # fallback: just take this one (rare)
                    i += 1
            else:
                break   # not enough remaining slots for a pair
    
    # Build Lambda_k
    Lambda_k = np.zeros_like(eigenvalues, dtype=complex)
    for idx in selected:
        Lambda_k[idx] = eigenvalues[idx]
    Lambda_k = np.diag(Lambda_k)
    
    Q = eigenvectors
    Q_inv = np.linalg.inv(Q)
    
    Ak = Q @ Lambda_k @ Q_inv
    return np.real(Ak)

def svd_reconstruct(A, k):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_k = np.zeros_like(S)
    S_k[:k] = S[:k]
    Ak = U @ np.diag(S_k) @ VT
    return Ak

# ====================== SQUARE IMAGE ======================
print("="*60)
print("SQUARE IMAGE")
print("="*60)

A = load_grayscale(SQUARE_IMAGE)
n = min(A.shape)
print(f"Image shape: {A.shape}")

print("\n----- EVD -----")
for k in K_VALUES:
    Ak = evd_reconstruct(A, k)
    err_img = np.abs(A - Ak)
    f_err = frobenius(A, Ak)
    print(f"k = {k:3d}   Frobenius Error = {f_err:.4f}")
    save_triple(A, Ak, err_img, f"EVD k={k}", f"square_EVD_k{k}.png")

print("\n----- SVD -----")
for k in K_VALUES:
    Ak = svd_reconstruct(A, k)
    err_img = np.abs(A - Ak)
    f_err = frobenius(A, Ak)
    print(f"k = {k:3d}   Frobenius Error = {f_err:.4f}")
    save_triple(A, Ak, err_img, f"SVD k={k}", f"square_SVD_k{k}.png")

# Full error curves
print("\nCalculating full error curves (this may take 1-2 minutes)...")
evd_errors = []
svd_errors = []

for k in range(1, n+1):
    Ak_evd = evd_reconstruct(A, k)
    Ak_svd = svd_reconstruct(A, k)
    evd_errors.append(frobenius(A, Ak_evd))
    svd_errors.append(frobenius(A, Ak_svd))
    if k % 50 == 0:
        print(f"  processed k = {k}/{n}")

# Plot EVD error curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, n+1), evd_errors, label="EVD")
plt.xlabel("k (number of components)")
plt.ylabel("Frobenius Error E(k)")
plt.title("EVD Reconstruction Error vs k")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "error_curve_EVD.png"), dpi=150, bbox_inches='tight')
plt.close()

# Plot SVD error curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, n+1), svd_errors, label="SVD", color='orange')
plt.xlabel("k (number of components)")
plt.ylabel("Frobenius Error E(k)")
plt.title("SVD Reconstruction Error vs k")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "error_curve_SVD.png"), dpi=150, bbox_inches='tight')
plt.close()

# Combined comparison plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, n+1), evd_errors, label="EVD")
plt.plot(range(1, n+1), svd_errors, label="SVD")
plt.xlabel("k")
plt.ylabel("Frobenius Error")
plt.title("EVD vs SVD Reconstruction Error")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "error_curve_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

print("Square image done.")

# ====================== RECTANGULAR IMAGE ======================
print("\n" + "="*60)
print("RECTANGULAR IMAGE (SVD only)")
print("="*60)

B = load_grayscale(RECT_IMAGE)
m = min(B.shape)
print(f"Image shape: {B.shape}")

print("\n----- SVD -----")
for k in K_VALUES:
    if k > m:
        print(f"k = {k} skipped (larger than min dimension {m})")
        continue
    Ak = svd_reconstruct(B, k)
    err_img = np.abs(B - Ak)
    f_err = frobenius(B, Ak)
    print(f"k = {k:3d}   Frobenius Error = {f_err:.4f}")
    save_triple(B, Ak, err_img, f"Rect SVD k={k}", f"rect_SVD_k{k}.png")

# Full error curve for rectangular
print("\nCalculating full error curve for rectangular image...")
rect_errors = []
for k in range(1, m+1):
    Ak = svd_reconstruct(B, k)
    rect_errors.append(frobenius(B, Ak))
    if k % 50 == 0:
        print(f"  processed k = {k}/{m}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, m+1), rect_errors, color='green')
plt.xlabel("k")
plt.ylabel("Frobenius Error E(k)")
plt.title("SVD Reconstruction Error (Rectangular Image)")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "error_curve_rect_SVD.png"), dpi=150, bbox_inches='tight')
plt.close()

print("\n==================== ALL DONE ====================")
print(f"All results are saved inside the folder: {OUTPUT_DIR}/")
print("You should see these files:")
print("  - square_EVD_k10.png, square_EVD_k50.png, square_EVD_k100.png")
print("  - square_SVD_k10.png, square_SVD_k50.png, square_SVD_k100.png")
print("  - rect_SVD_k10.png, rect_SVD_k50.png, rect_SVD_k100.png")
print("  - error_curve_EVD.png")
print("  - error_curve_SVD.png")
print("  - error_curve_comparison.png")
print("  - error_curve_rect_SVD.png")
