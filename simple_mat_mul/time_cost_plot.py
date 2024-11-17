#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Data
matrix_sizes = [256, 512, 1024, 2048]  # Common matrix sizes
slow_times = [0.12, 1.50, 15.13, 194.45]
fast_times = [0.08, 0.64, 5.05, 40.31]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, slow_times, 'ro-', label='Slow (ijk)', linewidth=2)
plt.plot(matrix_sizes, fast_times, 'bo-', label='Fast (kij)', linewidth=2)

# Customize the plot
plt.xlabel('Matrix Size (n × n)')
plt.ylabel('Time (seconds)')
plt.title('Matrix Multiplication Performance: Slow (ijk) vs Fast (kij)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Set exact values for x-axis ticks
plt.xticks(matrix_sizes, matrix_sizes)

# Add value annotations
for i, (slow, fast) in enumerate(zip(slow_times, fast_times)):
    plt.annotate(f'{slow:,.2f}s', (matrix_sizes[i], slow), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'{fast:,.2f}s', (matrix_sizes[i], fast), 
                textcoords="offset points", xytext=(0,-15), ha='center')

plt.tight_layout()
plt.show()
