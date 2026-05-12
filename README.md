# SVD-Based Image Compression Using Linear Algebra

A Python-based image processing project that demonstrates how **Singular Value Decomposition (SVD)** can be used for image compression through low-rank matrix approximation. The project converts grayscale images into numerical matrices, applies SVD, reconstructs compressed images using selected singular values, and evaluates the quality of compression using visual comparisons and PSNR.

---

## Project Overview

Digital images can be represented as matrices, where each matrix value corresponds to the brightness intensity of a pixel. This project uses this mathematical representation to compress images using **Singular Value Decomposition**, a fundamental technique in linear algebra.

Instead of storing the full image matrix, the image is decomposed into singular components. By keeping only the most important singular values, the image can be reconstructed with significantly less data while preserving most of the visual information.

The final implementation uses **rank k = 50**, which provides a strong balance between compression and image quality.

---

## What This Project Demonstrates

This project demonstrates:

- Image representation as numerical matrices
- Singular Value Decomposition for matrix factorization
- Low-rank approximation for image compression
- Image reconstruction using selected singular values
- Compression-quality trade-off analysis
- PSNR-based image quality evaluation
- Rank progression visualization
- Singular value decay and cumulative energy analysis

---

## Project Category

**Computer Vision | Image Processing | Linear Algebra | Numerical Computing**

Although this project does not train a machine learning model, it is closely related to **Computer Vision** because it works directly with image data, pixel matrices, reconstruction quality, and image transformation techniques.

---

## Technologies Used

- Python
- NumPy
- Pillow
- Matplotlib
- Linear Algebra
- Singular Value Decomposition
- Image Processing

---

## Mathematical Background

A grayscale image can be represented as a 2D matrix:

```text
A = image matrix
```

Each value in matrix `A` represents the brightness of one pixel.

Singular Value Decomposition decomposes the image matrix as:

```text
A = UΣVᵀ
```

Where:

- `U` contains the left singular vectors
- `Σ` contains singular values
- `Vᵀ` contains the right singular vectors

The singular values represent the importance of different image patterns. Larger singular values capture the most important visual information, while smaller values usually represent fine details or noise.

For compression, only the top `k` singular values are retained:

```python
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

Here, `A_k` is the reconstructed compressed image.

---

## How SVD Compression Works

The compression process follows these steps:

1. Load the input image.
2. Convert the image to grayscale.
3. Convert the image into a NumPy matrix.
4. Apply Singular Value Decomposition.
5. Select the top `k` singular values.
6. Reconstruct the image using low-rank approximation.
7. Save the compressed image.
8. Compare original and compressed images.
9. Evaluate compression quality using PSNR.
10. Generate visualizations for rank progression and singular value analysis.

---

## Key Features

- Reads and processes grayscale image files
- Applies SVD using NumPy
- Performs image reconstruction at multiple rank values
- Compresses images using low-rank approximation
- Calculates percentage of data used and data saved
- Evaluates image quality using PSNR
- Generates original vs. compressed image comparison
- Generates rank progression visualization
- Generates singular value and cumulative energy plots

---

## Results

At rank `k = 50`, the compressed image used only **19.6%** of the original data while saving **80.4%** of the storage requirement.

The project tested multiple rank values to observe the relationship between compression level and image quality.

| Rank k | Data Used | Data Saved | Quality Observation |
|--------|-----------|------------|--------------------|
| 1      | 0.4%      | 99.6%      | Very poor, highly blurred |
| 5      | 2.0%      | 98.0%      | Low detail, major distortion |
| 10     | 3.9%      | 96.1%      | Basic structure visible |
| 20     | 7.8%      | 92.2%      | Recognizable image |
| 50     | 19.6%     | 80.4%      | Clear image with strong compression |
| 100    | 39.1%     | 60.9%      | Near-original visual quality |

---

## PSNR Evaluation

Peak Signal-to-Noise Ratio, or PSNR, was used to measure the quality of reconstructed images.

A higher PSNR value means the compressed image is closer to the original image.

Example results from the project:

| Image Type | Rank k | Data Used | Data Saved | PSNR |
|-----------|--------|-----------|------------|------|
| Grayscale portrait image | 50 | 19.6% | 80.4% | 32.1 dB |
| Structured test image | 50 | 19.6% | 80.4% | 51.0 dB |

These results show that SVD compression can preserve strong visual quality while significantly reducing the amount of stored image data.

---

## Rank Progression Analysis

The image was reconstructed at different rank values:

```text
k = 1, 5, 10, 20, 50, 100
```

The progression shows how image quality improves as more singular values are retained.

- At very low ranks, only the most dominant structure is visible.
- At medium ranks, the image becomes recognizable.
- At rank `k = 50`, the image keeps most important visual details.
- At rank `k = 100`, the reconstructed image becomes very close to the original.

This demonstrates the trade-off between compression and reconstruction quality.

---

## Singular Value Analysis

The project also analyzes singular values to understand why SVD compression works.

The singular values usually decay rapidly, meaning the most important image information is concentrated in the first few components. This allows the image to be reconstructed using only a limited number of singular values.

The project generates:

- Singular value decay plot
- Cumulative energy captured plot

These plots help explain how much visual information is preserved at different rank values.

---

## Project Structure

```text
SVD-Image-Compression-Using-Linear-Algebra/
│
├── README.md
├── requirements.txt
├── svd_image_compression.py
│
├── images/
│   ├── input_image.jpg
│   └── face-input2.jpg
│
├── outputs/
│   ├── compressed_image.jpg
│   ├── original_vs_compressed.png
│   ├── rank_progression.png
│   └── singular_value_analysis.png
│
└── report/
    └── SVD_Image_Compression_Report.pdf
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/bilalfayyaz66/SVD-Image-Compression-Using-Linear-Algebra.git
```

Move into the project folder:

```bash
cd SVD-Image-Compression-Using-Linear-Algebra
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file with the following dependencies:

```text
numpy
pillow
matplotlib
```

---

## How to Run

Run the Python script:

```bash
python svd_image_compression.py
```

The program will load the image, apply SVD compression, save the compressed output, and generate visual comparison plots.

---

## Example Code Snippet

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("images/input_image.jpg").convert("L")
A = np.array(image, dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

k = 50
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

compressed_image = np.clip(A_k, 0, 255).astype(np.uint8)
Image.fromarray(compressed_image).save("outputs/compressed_image.jpg")
```

---

## Output Files

The project generates the following output files:

| Output File | Description |
|------------|-------------|
| `compressed_image.jpg` | Final SVD-compressed image |
| `original_vs_compressed.png` | Side-by-side comparison of original and compressed image |
| `rank_progression.png` | Image reconstruction at different rank values |
| `singular_value_analysis.png` | Singular value decay and cumulative energy plots |

---

## Example Output Summary

```text
SVD Image Compression Completed

Image size: 512 x 512
Original data values: 262,144
Selected rank: k = 50
Compressed data used: 19.6%
Data saved: 80.4%
PSNR: 32.1 dB
```

---

## Skills Demonstrated

This project demonstrates practical knowledge of:

- Computer Vision fundamentals
- Image processing
- Linear algebra
- Matrix decomposition
- Singular Value Decomposition
- Low-rank approximation
- Image compression
- Image reconstruction
- Numerical computing with NumPy
- PSNR-based quality evaluation
- Data visualization using Matplotlib
- Scientific programming in Python

---

## Real-World Relevance

SVD and low-rank approximation are widely used in areas such as:

- Image compression
- Noise reduction
- Dimensionality reduction
- Recommendation systems
- Computer vision preprocessing
- Pattern extraction
- Data compression
- Feature reduction

This project shows how mathematical techniques can be applied to solve real-world image processing problems.

---

## Future Improvements

Possible future improvements include:

- Extend compression to RGB color images
- Add a Streamlit web interface
- Allow users to upload custom images
- Compare SVD compression with JPEG compression
- Add SSIM as another image quality metric
- Support batch image compression
- Add interactive rank selection
- Export compressed results in multiple formats

---

## Conclusion

This project successfully demonstrates image compression using Singular Value Decomposition. By representing images as matrices and reconstructing them using only the most important singular values, the project achieves significant data reduction while preserving visual quality.

At rank `k = 50`, the compressed image uses only **19.6%** of the original data and saves **80.4%**, while still maintaining strong visual similarity to the original image. This highlights the power of linear algebra in computer vision and image processing applications.

---

## Author

**Bilal Fayyaz**  
BS Data Science  
FAST NUCES, Islamabad  
GitHub: [github.com/bilalfayyaz66](https://github.com/bilalfayyaz66)

---

## Repository Topics

```text
python
numpy
pillow
matplotlib
computer-vision
image-processing
linear-algebra
svd
singular-value-decomposition
image-compression
low-rank-approximation
matrix-factorization
psnr
```
