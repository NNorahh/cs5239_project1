#!/bin/bash

# Install required packages
apt install -y make gcc msr-tools

# Compile the test program
make

# Task B: Unlock RDPMC
sudo insmod hack_cr4.ko

# Task C: Enable PMC0 by configuring MSR IA32_PERF_GLOBAL_CTRL(0x38f)
# Load the MSR driver
sudo modprobe msr

# Read and verify MSR content (expected: 70000000f)
sudo rdmsr -a 0x38f

MSR_VALUE=$(sudo rdmsr -a 0x38f)
if [ "$MSR_VALUE" != "70000000f" ]; then
    echo "Configuring MSR IA32_PERF_GLOBAL_CTRL to 0x70000000f"
    sudo wrmsr -a 0x38f 0x70000000f
else
    echo "MSR IA32_PERF_GLOBAL_CTRL already correctly configured"
fi

# Task D: Enable L2_RQSTS.MISS monitoring via IA32_PERFEVENTSEL0(PMC0)
# Verify PMC0 status
sudo rdmsr -a 0x186
# Clear IA32_PMC0
sudo wrmsr -a 0xc1 0x00
# Configure PMC0 for L2_RQSTS.MISS
sudo wrmsr -a 0x186 0x413f24

# Run the test program on CPU core 0
taskset 0x01 ./mat_mul 1024
