# Matrix Multiplication: CPU vs GPU Performance Analysis 🚀

> Comprehensive benchmark analysis comparing CPU (NumPy) vs GPU (CuPy) performance for matrix multiplication operations.


---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Features](#features)
- [Quick Start](#quick-start)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements and analyzes the performance of matrix multiplication algorithms on **CPU** (using NumPy) versus **GPU** (using CuPy). The benchmark covers matrix sizes from **100×100** to **4000×4000**, providing insights into:

- Execution time comparison
- Speedup factors
- Computational throughput (GFLOPS)
- Scaling behavior
- Optimal use cases for each platform

**Category:** Parallel & GPU Computing  
**Algorithm:** Matrix Multiplication  
**Framework:** Python, NumPy, CuPy, Google Colab

---

## 🏆 Key Findings

### 📊 Performance Summary

| Metric | Value | Note |
|--------|-------|------|
| **Peak GPU Speedup** | **29.1×** | @ 1500×1500 |
| **GPU Throughput** | **2,792 GFLOPS** | @ 4000×4000 |
| **GPU Efficiency** | **34.5%** | Of theoretical peak (Tesla T4) |
| **Crossover Point** | **~250-500** | Below: CPU competitive, Above: GPU dominates |
| **All Tests Validated** | **✅ 10/10** | Max error < 1e-3 |

### 🔍 Key Insights

1. **GPU slower for small matrices** (100×100: 0.24× speedup)
   - Overhead dominates for matrices < 500×500
   
2. **Optimal GPU range**: 1500-4000×4000
   - Consistent speedup: 20-29×
   
3. **GPU throughput scales efficiently**
   - 10 GFLOPS → 2,792 GFLOPS
   - CPU plateaus at ~130 GFLOPS

---

## ✨ Features

- ✅ **Comprehensive Benchmarking**: 10 matrix sizes (100-4000)
- ✅ **Statistical Rigor**: Multiple iterations with standard deviation
- ✅ **Validation**: CPU vs GPU result verification (floating-point tolerance)
- ✅ **Professional Visualizations**: 4 informative plots
  - Execution time (log-log scale)
  - Speedup factor analysis
  - Side-by-side comparison
  - Throughput (GFLOPS)
- ✅ **Detailed Analysis**: Theory, implementation, and performance insights
- ✅ **Google Colab Ready**: One-click execution with free GPU

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Runtime → Change runtime type → **GPU**
3. Runtime → Run all
4. Wait 5-10 minutes for results

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/Master-Megatron/matrix-multiplication-cpu-vs-gpu.git
cd matrix-multiplication-cpu-vs-gpu

# Install dependencies (requires CUDA-capable GPU)
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/Matrix_Multiplication_FINAL.ipynb
```

---

## 📊 Results

### Execution Time Comparison

![Execution Time](results/execution_time_plot.png)

Matrix 4000×4000:
- **CPU**: 1.186 seconds
- **GPU**: 0.046 seconds
- **Speedup**: 25.88×

### Speedup Pattern (M-Shape)

The benchmark reveals an interesting **M-shape speedup pattern**:
```
Phase 1: GPU overhead (100-500)      → Low speedup (0.24-6.18×)
Phase 2: Growth (750-1500)           → Rapid increase (8.71-29.10×)
Phase 3: Peak (1500-2000)            → Maximum efficiency (29×)
Phase 4: Minor drop (2500-3000)      → CPU variance effect (20-22×)
Phase 5: Recovery (4000)             → Stabilization (25.88×)
```

### Throughput Analysis

| Platform | Peak Performance | Efficiency |
|----------|------------------|------------|
| **GPU** | 2,792 GFLOPS | 34.5% (excellent) |
| **CPU** | ~130 GFLOPS | 10-15% (typical) |

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for local execution)
- Google Colab account (for cloud execution)

### Dependencies
```bash
numpy>=1.21.0
cupy-cuda12x>=12.0.0  # Adjust for your CUDA version
matplotlib>=3.5.0
pandas>=1.3.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### Running the Benchmark
```python
# Basic usage (in notebook)
import numpy as np
import cupy as cp

# Define matrix sizes to test
MATRIX_SIZES = [100, 500, 1000, 2000, 4000]
N_ITERATIONS = 3

# Run benchmark (see notebook for full code)
# Results will be automatically visualized
```

### Customizing the Benchmark
```python
# Change matrix sizes
MATRIX_SIZES = [500, 1000, 2000, 3000]

# Increase iterations for better accuracy
N_ITERATIONS = 5

# Change data type
DTYPE = np.float64  # Default: float32
```

---

## 📂 Project Structure
```
matrix-multiplication-cpu-vs-gpu/
│
├── notebooks/
│   └── Matrix_Multiplication_FINAL.ipynb  # Main notebook
│
├── results/
│   ├── benchmark_visualization.png        # Performance graphs
│   ├── benchmark_results.csv              # Raw data
│   └── analysis_summary.txt               # Text summary
│
├── docs/
│   ├── theory.md                          # Theoretical background
│   ├── setup_guide.md                     # Setup instructions
│   └── architecture_comparison.md         # CPU vs GPU architecture
│
├── images/
│   └── [visualization assets]
│
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
└── .gitignore                             # Git ignore rules
```

---

## 🔬 Methodology

### Benchmark Design

1. **Matrix Generation**: Random matrices using NumPy's `randn()`
2. **Warm-up Runs**: Initial execution to warm caches
3. **Multiple Iterations**: 3 runs per size for statistical reliability
4. **Synchronization**: Proper GPU synchronization for accurate timing
5. **Validation**: Numerical comparison within floating-point tolerance (1e-3)

### Metrics Collected

- **Execution Time**: Wall-clock time for matrix multiplication
- **Speedup Factor**: Ratio of CPU time to GPU time
- **GFLOPS**: Floating-point operations per second
- **Error Statistics**: Maximum absolute error between CPU/GPU results

### Hardware Used

- **CPU**: Colab CPU (varies: Intel Xeon, AMD EPYC)
- **GPU**: NVIDIA Tesla T4 (8.1 TFLOPS FP32 peak)
- **Memory**: System RAM + 15GB GPU memory

---

## 🎓 Theoretical Background

### Matrix Multiplication Complexity

For N×N matrices:
- **Operations**: 2N³ FLOPS (N³ multiplications + N³ additions)
- **Complexity**: O(N³)
- **Example**: 4000×4000 requires 128 billion operations

### CPU Architecture

- **Design**: Optimized for sequential processing
- **Cores**: 4-16 (typical desktop)
- **Strengths**: Low latency, complex branching
- **Limitations**: Limited parallelism

### GPU Architecture

- **Design**: Massively parallel (SIMT)
- **Cores**: 2,560 CUDA cores (Tesla T4)
- **Strengths**: High throughput, data parallelism
- **Limitations**: High overhead for small tasks

### Why GPU is Faster?

1. **Parallelism**: Thousands of threads compute elements simultaneously
2. **Memory Bandwidth**: 320 GB/s (GPU) vs 50 GB/s (CPU)
3. **Specialized Hardware**: Tensor cores, optimized BLAS libraries

---

## 📈 Performance Guidelines

### When to Use GPU

✅ Matrix size ≥ 1000×1000  
✅ Multiple matrix operations (batch processing)  
✅ Training machine learning models  
✅ Scientific simulations  

### When to Use CPU

✅ Matrix size < 500×500  
✅ Single operation  
✅ Complex control flow  
✅ GPU not available  

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution

- [ ] Additional matrix sizes
- [ ] Different data types (int, complex)
- [ ] Sparse matrix support
- [ ] Multi-GPU benchmarking
- [ ] AMD GPU (ROCm) support
- [ ] Detailed profiling (memory bandwidth, cache hits)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Your Name**
- GitHub: [@Master-Megatron](https://github.com/Master-Megatron)
- Email: megatronelectronicmail@gmail.com


---

## 🙏 Acknowledgments

- **NumPy Team**: For excellent CPU linear algebra library
- **CuPy Team**: For GPU-accelerated NumPy interface
- **NVIDIA**: For CUDA platform and cuBLAS library
- **Google Colab**: For providing free GPU access

---

## 📚 References

1. [NumPy Documentation](https://numpy.org/doc/)
2. [CuPy Documentation](https://docs.cupy.dev/)
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
4. [Matrix Multiplication Algorithms](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)

---

## ⭐ If you found this helpful, please star this repository!

[![Star History Chart](https://api.star-history.com/svg?repos=Master-Megatron/matrix-multiplication-cpu-vs-gpu&type=Date)](https://star-history.com/#Master-Megatron/matrix-multiplication-cpu-vs-gpu&Date)

---

**Last Updated**: January 2025  
**Status**: ✅ Active
