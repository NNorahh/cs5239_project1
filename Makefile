# Create output directory if it doesn't exist
$(shell mkdir -p output)

build_opt_parallel:
	gcc -o output/matrix_mult_parallel_l2 optimization_parallel/matrix_mult_parallel_l2.c -pthread

build_opt_simd:
	gcc -o output/mat_mul_simd optimization_SIMD/mat_mul_simd.c -O2 -g -mavx2 -mavx512f

build_naive:
	gcc -o output/mat_mul_naive task4_hardware_counters/mat_mul_origin.c

build: build_opt_parallel build_opt_simd build_naive

clean:
	rm -rf output/*
