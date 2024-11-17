import matplotlib.pyplot as plt
import numpy as np

# Data for matrix sizes
matrix_sizes = ['512', '1024', '2048']

# L1 cache loads and misses for ijk and kij (normalized to percentages)
l1_loads = {
    'ijk': [2958801913, 23647988484, 189121292839],
    'kij': [2958594144, 23645538405, 189076485716]
}
l1_misses = {
    'ijk': [135102013, 1080347809, 13949132985],
    'kij': [17601764, 139988463, 1195638785]
}

# L2 cache loads (references) and misses for ijk and kij (normalized to percentages)
l2_loads = {
    'ijk': [135275630, 1081460664, 14670456209],
    'kij': [35252010, 279388213, 2315535987]
}
l2_misses = {
    'ijk': [133724908, 1078489514, 8693090760],
    'kij': [18250040, 141712329, 1116813844]
}

# Calculate miss ratios (as percentages)
l1_miss_ratio_ijk = [misses/loads*100 for misses, loads in zip(l1_misses['ijk'], l1_loads['ijk'])]
l1_miss_ratio_kij = [misses/loads*100 for misses, loads in zip(l1_misses['kij'], l1_loads['kij'])]

l2_miss_ratio_ijk = [misses/loads*100 for misses, loads in zip(l2_misses['ijk'], l2_loads['ijk'])]
l2_miss_ratio_kij = [misses/loads*100 for misses, loads in zip(l2_misses['kij'], l2_loads['kij'])]

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Set width of bars and positions of the bars
width = 0.35
x = np.arange(len(matrix_sizes))

# Create bars for L1 cache miss ratios
ax1.bar(x - width/2, l1_miss_ratio_ijk, width, label='ijk', color='skyblue')
ax1.bar(x + width/2, l1_miss_ratio_kij, width, label='kij', color='lightcoral')
ax1.set_title('L1 Cache Miss Ratio')
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Miss Ratio (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(matrix_sizes)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Create bars for L2 cache miss ratios
ax2.bar(x - width/2, l2_miss_ratio_ijk, width, label='ijk', color='skyblue')
ax2.bar(x + width/2, l2_miss_ratio_kij, width, label='kij', color='lightcoral')
ax2.set_title('L2 Cache Miss Ratio')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Miss Ratio (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(matrix_sizes)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

