import matplotlib.pyplot as plt
import numpy as np

# Data for matrix sizes
matrix_sizes = ['512', '1024', '2048']

# L1 cache loads and misses for ijk and kij (normalized to percentages)
l1_loads = {
    'ijk': [2958874388, 23648715259, 189122813166],
    'kij': [2958592729, 23645761803, 189077345014]
}
l1_misses = {
    'ijk': [134996042, 1084343286, 13938277896],
    'kij': [17598133, 143784185, 1216271014]
}

# LLC cache loads (references) and misses for ijk and kij (normalized to percentages)
llc_loads = {
    'ijk': [131577155, 1075390810, 8669856942],
    'kij': [306639, 1608774, 8230791]
}
llc_misses = {
    'ijk': [80963, 124675275, 4185222822],
    'kij': [13861, 588541, 5402496]
}

# Set up the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Set width of bars and positions
width = 0.35
x = np.arange(len(matrix_sizes))

# Plot L1 cache misses (using log scale) - REVERSED COLORS
ax1.bar(x - width/2, l1_misses['ijk'], width, label='ijk', color='lightcoral')
ax1.bar(x + width/2, l1_misses['kij'], width, label='kij', color='skyblue')
ax1.set_yscale('log')
ax1.set_title('L1 Cache Misses Comparison')
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Number of Cache Misses')
ax1.set_xticks(x)
ax1.set_xticklabels(matrix_sizes)
ax1.legend()

# Add value labels on top of each bar
for i, v in enumerate(l1_misses['ijk']):
    ax1.text(i - width/2, v, f'{v:,.0f}', ha='center', va='bottom', rotation=0)
for i, v in enumerate(l1_misses['kij']):
    ax1.text(i + width/2, v, f'{v:,.0f}', ha='center', va='bottom', rotation=0)

# Plot LLC cache misses (using log scale) - REVERSED COLORS
ax2.bar(x - width/2, llc_misses['ijk'], width, label='ijk', color='lightcoral')
ax2.bar(x + width/2, llc_misses['kij'], width, label='kij', color='skyblue')
ax2.set_yscale('log')
ax2.set_title('LLC Cache Misses Comparison')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Number of Cache Misses')
ax2.set_xticks(x)
ax2.set_xticklabels(matrix_sizes)
ax2.legend()

# Add value labels on top of each bar
for i, v in enumerate(llc_misses['ijk']):
    ax2.text(i - width/2, v, f'{v:,.0f}', ha='center', va='bottom', rotation=0)
for i, v in enumerate(llc_misses['kij']):
    ax2.text(i + width/2, v, f'{v:,.0f}', ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.show()

