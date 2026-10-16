# GPU-Accelerated 2D Convolution for High-Performance AI Workloads

## Overview

This project implements and evaluates 2D convolution using CPU and NVIDIA CUDA GPU computing. It explores GPU parallelism and shared-memory optimization while analyzing performance across different input image sizes.

2D convolution is a fundamental operation in image processing and Convolutional Neural Networks (CNNs). Accelerating this operation helps demonstrate how parallel computing can improve computational performance in AI and high-performance computing workloads.

## Objectives

- Implement 2D convolution using a CPU-based approach.
- Develop a basic CUDA implementation using GPU threads.
- Explore shared-memory optimization.
- Validate GPU outputs against the CPU reference.
- Benchmark execution time and calculate speedup.
- Visualize convolution-based edge detection.

## Technologies Used

- C++ and CUDA
- NVIDIA Tesla T4 GPU
- Google Colab
- NumPy and Matplotlib
- CUDA events for GPU performance measurement

## Implementation

The project evaluates three implementations:

1. **CPU Convolution:** A compiled C++ implementation using optimization flag `-O3`.
2. **Basic CUDA Convolution:** Each GPU thread computes one output pixel using global memory.
3. **Shared-Memory CUDA Convolution:** A tiled implementation that uses shared memory to reuse input data between threads.

A 3 × 3 edge-detection kernel is applied to images of different sizes.

## Experimental Results

| Image Size | CPU (ms) | Basic CUDA (ms) | Shared CUDA (ms) | Basic CUDA Speedup |
|---|---:|---:|---:|---:|
| 512 × 512 | 0.203 | 0.016380 | 0.034100 | 12.39× |
| 1024 × 1024 | 0.810 | 0.061588 | 0.111614 | 13.15× |
| 2048 × 2048 | 3.469 | 0.233882 | 0.320492 | 14.83× |
| 4096 × 4096 | 18.4269 | 0.694127 | 1.749280 | 26.55× |

*Note:* CPU timings are from compiled C++ with `-O3`. GPU timings measure kernel execution and exclude host-device data transfers. Results are specific to the tested environment and workload.

## Key Findings

- The basic CUDA implementation achieved a measured speedup of approximately **12.39× to 26.55×** over the CPU implementation.
- Both CUDA implementations matched the CPU reference, with a maximum absolute difference of `0.0`.
- The shared-memory implementation was slower than the basic CUDA implementation for the tested 3 × 3 kernel.
- Shared-memory optimization does not guarantee better performance; synchronization and tile-loading overhead can outweigh data-reuse benefits for some workloads.

## Future Enhancements

- Experiment with different convolution kernel sizes.
- Tune CUDA block dimensions.
- Explore constant memory and additional memory optimizations.
- Extend the implementation to RGB images and larger workloads.
- Investigate integration into CNN-based image-processing pipelines.
