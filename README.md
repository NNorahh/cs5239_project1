# cs5239_project1
# Project Name

## Overview
This repository contains various modules for optimizing computational tasks, including SIMD optimizations, cache optimizations, parallelization, and hardware counter tracking. Additionally, the project includes basic implementations like simple matrix multiplication for testing purposes.

## Project Structure

```
.
├── Makefile                    # Build automation tool to compile the project.
├── README.md                   # Project documentation file.
├── combined_vis                # Directory containing combined visualizations of optimization results.
├── comparison                  # Folder for comparison scripts of optimization strategies.
├── optimization_SIMD           # SIMD optimization code.
├── optimization_cache          # Cache optimization code.
├── optimization_compiler       # Compiler optimization code.
├── optimization_parallel       # Parallel computing optimization code.
├── optimization_parallel&simd  # Combination of parallel computing and SIMD optimizations.
├── simple_mat_mul              # Basic matrix multiplication implementation for benchmarking.
├── task3_zip                   # Task 3 code.
└── task4_hardware_counters     # Task 4 hardware counter code.

```



## Environment

### Required Dependencies
- **C++ Compiler**: Ensure that you have a C++ compiler (e.g., GCC, Clang) installed for building the project.
- **Make**: The project uses `Make` to automate the build process.
- **Hardware Counters**: For tasks related to hardware counters, ensure that your system supports monitoring hardware performance events (e.g., using `perf` on Linux).
  
### System Requirements
- **Operating System**: Linux-based systems recommended, especially for hardware counter tracking.
- **Libraries**: The project may require additional libraries based on specific optimization tasks (e.g., `OpenMP` for parallelism, `SIMD` instructions for vectorization).

