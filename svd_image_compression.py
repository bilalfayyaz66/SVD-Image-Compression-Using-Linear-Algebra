#SVD Image Compression
#Requirement for running this: pip install numpy pillow matplotlib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import sys
sys.stdout.reconfigure(encoding='utf-8')


#Configuration

FINAL_RANK = 50

#User Input of Image

print("SVD Image Compression")
print("-" * 30)

user_input = input("Enter image filename (or press Enter for default 'input_image.jpg'): ").strip()
IMAGE_PATH = user_input if user_input else "input_image.jpg"



#Core Functions

def svd_compress(U, S, Vt, k):
    # Reconstructing the image using top-k singular values: Ak = Uk @ diag(Sk) @ Vtk
    return np.clip(U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], 0, 255)


def compression_ratio(m, n, k):
    # Returning percentage of original data used at rank k
    return (k * (m + n + 1)) / (m * n) * 100


def psnr(original, compressed):
    # Peak Signal-to-Noise Ratio in dB. Higher means better quality
    mse = np.mean((original - compressed) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255 / np.sqrt(mse))


#Loading image

if not os.path.exists(IMAGE_PATH):
    print(f"Error: '{IMAGE_PATH}' not found. Place your image in the same folder.")
    sys.exit(1)

image = Image.open(IMAGE_PATH).convert('L')
A = np.array(image, dtype=np.float64)
m, n = A.shape

print(f"Image loaded  : {IMAGE_PATH}  ({m} x {n} pixels)")


#Applying SVD technique

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"SVD computed  : {len(S)} singular values")
print(f"Compressing   : rank k = {FINAL_RANK}")


#Reconstructing at Final Rank

A_compressed = svd_compress(U, S, Vt, FINAL_RANK)
comp_pct     = compression_ratio(m, n, FINAL_RANK)
quality      = psnr(A, A_compressed)

print(f"Data used     : {comp_pct:.1f}%  (saved {100 - comp_pct:.1f}%)")
print(f"PSNR          : {quality:.1f} dB")


#Figure 1: Original vs Compressed

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SVD Image Compression", fontsize=14, fontweight='bold')

axes[0].imshow(A, cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f"Original\n{m} x {n}  |  Full rank = {len(S)}", fontsize=11)
axes[0].axis('off')

axes[1].imshow(A_compressed, cmap='gray', vmin=0, vmax=255)
axes[1].set_title(
    f"Compressed  (k = {FINAL_RANK})\n"
    f"Data used: {comp_pct:.1f}%  |  PSNR: {quality:.1f} dB",
    fontsize=11
)
axes[1].axis('off')

plt.tight_layout()
plt.savefig("output_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved         : output_comparison.png")


#Figure 2: Rank Progression 
ranks = [k for k in [1, 5, 10, 20, 50, 100] if k <= len(S)]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("SVD Compression — Rank Progression", fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, k in enumerate(ranks):
    A_k  = svd_compress(U, S, Vt, k)
    comp = compression_ratio(m, n, k)
    q    = psnr(A, A_k)
    axes[i].imshow(A_k, cmap='gray', vmin=0, vmax=255)
    axes[i].set_title(f"Rank k = {k}\n{comp:.1f}% data  |  PSNR: {q:.1f} dB", fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("output_rank_progression.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved         : output_rank_progression.png")


#Figure 3: Singular Value Analysis

top = min(100, len(S))
cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Singular Value Analysis", fontsize=14, fontweight='bold')

axes[0].plot(range(1, top + 1), S[:top], color='steelblue', linewidth=2)
axes[0].set_xlabel("Rank k")
axes[0].set_ylabel("Singular Value")
axes[0].set_title("Singular Values (Top 100)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, top + 1), cumulative_energy[:top], color='darkorange', linewidth=2)
axes[1].axhline(90, color='gray', linestyle='--', linewidth=1, label='90%')
axes[1].axhline(99, color='red',  linestyle='--', linewidth=1, label='99%')
axes[1].set_xlabel("Rank k")
axes[1].set_ylabel("Cumulative Energy (%)")
axes[1].set_title("Cumulative Energy Captured")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output_singular_values.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved         : output_singular_values.png")


#Saving Final Compressed Image

output_array = np.clip(A_compressed, 0, 255).astype(np.uint8)
Image.fromarray(output_array, mode='L').save("output_compressed.jpg", quality=95)
print("Saved         : output_compressed.jpg  <- final output image")

print("\nAll output files saved in the current directory:")
