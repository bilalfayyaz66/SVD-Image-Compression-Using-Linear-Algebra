# SVD-Based Image Compression Using Linear Algebra

This project demonstrates image compression using **Singular Value Decomposition (SVD)**, a core linear algebra technique used for matrix factorization and low-rank approximation. The system converts a grayscale image into a numerical pixel matrix, decomposes it using SVD, reconstructs the image using a limited number of singular values, and evaluates the quality of compression using PSNR.

## Project Overview

Digital images can be represented as matrices where each pixel corresponds to a numerical intensity value. Instead of storing every pixel directly, this project applies SVD to identify the most important structural patterns in the image and reconstructs the image using only the top `k` singular values.

The final implementation uses `k = 50`, which significantly reduces the amount of stored data while preserving the key visual structure of the original image.

## Key Features

- Converts grayscale images into 2D numerical matrices
- Applies Singular Value Decomposition using NumPy
- Performs low-rank image reconstruction at selected rank values
- Compares original and compressed images visually
- Generates rank progression outputs for `k = 1, 5, 10, 20, 50, 100`
- Calculates compression ratio and data saved percentage
- Evaluates reconstruction quality using Peak Signal-to-Noise Ratio (PSNR)
- Visualizes singular value decay and cumulative energy retention

## Technologies Used

- Python
- NumPy
- Pillow
- Matplotlib
- Linear Algebra
- Singular Value Decomposition
- Image Processing

## Mathematical Approach

An input grayscale image is represented as a matrix `A`, where each element corresponds to pixel brightness.

SVD decomposes the image matrix as:

```text
A = UΣVᵀ
