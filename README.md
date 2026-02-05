
# FlashAttention-Metal
This project implements a high-performance FlashAttention kernel on Apple Silicon using MLX and Metal. By applying tiling and online softmax techniques, it optimizes memory IO and computes exact attention without materializing the massive $N \times N$ matrix. Designed for M-series GPUs, this implementation leverages threadgroup memory (SRAM) to achieve significant speedups and memory efficiency compared to standard attention mechanisms as sequence lengths scale.
