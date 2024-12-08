import numpy as np
import cv2
import pywt
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

# Load the image
img = cv2.imread('/kaggle/input/lanapic/The-recovered-test-image-Elaine-obtained-from-an-authorized-set-of-four-shadow-images-one.ppm.png', cv2.IMREAD_GRAYSCALE)

# ---------------- 1. Apply DWT to the Image ---------------- #
coeffs = pywt.dwt2(img, 'haar')  # Single-level DWT using 'haar' wavelet
cA, (cH, cV, cD) = coeffs  # Approximation, Horizontal, Vertical, Diagonal components

# ---------------- 2. Apply DCT to the Approximation Image ---------------- #
dct_img = dct(dct(cA.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

# Set 50% of the DCT coefficients to zero (reduce compression loss)
n = dct_img.size
nz = int(0.5 * n)  # Keep 50% of coefficients
indices = np.unravel_index(np.argsort(-np.abs(dct_img), axis=None), dct_img.shape)
zero_mask = np.zeros_like(dct_img, dtype=bool)
zero_mask[indices[0][nz:], indices[1][nz:]] = True
dct_img[zero_mask] = 0

# ---------------- 3. Create the Chaotic Map ---------------- #a
chaotic_mat = np.zeros_like(dct_img)
x0, y0, z0 = 0.2, 0.4, 0.6
a = 10
for i in range(chaotic_mat.size):
    x = np.mod(np.sin(a * y0) + np.cos(a * z0), 1)
    y = np.mod(np.sin(a * z0) + np.cos(a * x0), 1)
    z = np.mod(np.sin(a * x0) + np.cos(a * y0), 1)
    chaotic_mat.flat[i] = x + y + z
    x0, y0, z0 = x, y, z

# Scale chaotic map to match DCT range
chaotic_mat = (chaotic_mat - chaotic_mat.min()) / (chaotic_mat.max() - chaotic_mat.min())
chaotic_mat *= np.max(np.abs(dct_img))

# Encrypt the chaotic matrix using XOR encryption
encryption_key = 12345
chaotic_mat_uint32 = chaotic_mat.astype(np.uint32)  # Convert chaotic map to integers
encrypted_mat_uint32 = np.bitwise_xor(chaotic_mat_uint32, encryption_key)
encrypted_mat = encrypted_mat_uint32.astype(np.float64)

# ---------------- 4. Encrypt the DCT Coefficients ---------------- #
scrambled_dct = np.copy(dct_img)
non_zero_mask = scrambled_dct != 0
scrambled_dct[non_zero_mask] *= encrypted_mat[non_zero_mask]

# ---------------- 5. Decrypt the Encrypted Image ---------------- #
decrypted_mat_uint32 = np.bitwise_xor(encrypted_mat_uint32, encryption_key)
decrypted_mat = decrypted_mat_uint32.astype(np.float64)

# Rescale the decrypted matrix to ensure it's within a valid range
decrypted_mat = np.clip(decrypted_mat, 0, np.max(dct_img))

# Unscramble the non-zero DCT coefficients using the decrypted chaotic matrix
reconstructed_dct = np.copy(scrambled_dct)
reconstructed_dct[non_zero_mask] /= decrypted_mat[non_zero_mask]

# Reconstruct the approximation coefficients using IDCT
reconstructed_cA = idct(idct(reconstructed_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
reconstructed_cA = np.clip(reconstructed_cA, 0, 255).astype(np.uint8)

# Perform inverse DWT to get the decrypted image
decrypted_img = pywt.idwt2((reconstructed_cA, (cH, cV, cD)), 'haar')
decrypted_img = np.clip(decrypted_img, 0, 255).astype(np.uint8)

# ---------------- Error Metrics ---------------- #
def calculate_metrics(original, compressed):
    # Resize the images to the same shape (if required)
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
        
    mse = np.mean((original - compressed) ** 2)
    rmse = sqrt(mse)
    psnr = 20 * log10(255 / sqrt(mse)) if mse != 0 else float('inf')
    ssim_index = ssim(original, compressed, data_range=compressed.max() - compressed.min())
    return mse, rmse, psnr, ssim_index

mse, rmse, psnr, ssim_index = calculate_metrics(img, decrypted_img)

# ---------------- Display Results ---------------- #
plt.figure(figsize=(15, 12))

# Original Image
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Histogram of Original Image
plt.subplot(3, 3, 2)
plt.hist(img.ravel(), bins=256, color='black')
plt.title('Histogram: Original Image')

# DWT Approximation
plt.subplot(3, 3, 3)
plt.imshow(cA, cmap='gray')
plt.title('DWT (Approximation)')
plt.axis('off')

# DCT Image
plt.subplot(3, 3, 4)
plt.imshow(np.log(1 + np.abs(dct_img)), cmap='gray')
plt.title('DCT Image')
plt.axis('off')

# Encrypted Image
plt.subplot(3, 3, 5)
plt.imshow(np.log(1 + np.abs(scrambled_dct)), cmap='gray')
plt.title('Encrypted DCT Image')
plt.axis('off')

# Histogram of Encrypted Image
plt.subplot(3, 3, 6)
plt.hist(scrambled_dct.ravel(), bins=256, color='red')
plt.title('Histogram: Encrypted DCT')

# Decrypted Image
plt.subplot(3, 3, 7)
plt.imshow(decrypted_img, cmap='gray')
plt.title('Decrypted Image')
plt.axis('off')

# Histogram of Decrypted Image
plt.subplot(3, 3, 8)
plt.hist(decrypted_img.ravel(), bins=256, color='blue')
plt.title('Histogram: Decrypted Image')

# Error Metrics Display
plt.subplot(3, 3, 9)
plt.text(0.1, 0.6, f"MSE: {mse:.4f}", fontsize=12)
plt.text(0.1, 0.5, f"RMSE: {rmse:.4f}", fontsize=12)
plt.text(0.1, 0.4, f"PSNR: {psnr:.2f} dB", fontsize=12)
plt.text(0.1, 0.3, f"SSIM: {ssim_index:.4f}", fontsize=12)
plt.axis('off')
plt.title('Error Metrics')

plt.tight_layout()
plt.show()

# Print Metrics
print("Error Metrics:")
print(f" - MSE: {mse:.4f}")
print(f" - RMSE: {rmse:.4f}")
print(f" - PSNR: {psnr:.2f} dB")
print(f" - SSIM: {ssim_index:.4f}")
