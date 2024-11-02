# Create output directory if it doesn't exist
$(shell mkdir -p output)

build_opt_parallel:
	gcc -o output/matrix_mult_parallel_l2 optimization_parallel/matrix_mult_parallel_l2.c -pthread

build_opt_simd:
	gcc -o output/mat_mul_simd optimization_SIMD/mat_mul_simd.c -O2 -g -mavx2 -mavx512f

build_opt_compiler:
	gcc -o output/mat_mul_ftree_vectorize optimization_compiler/mat_mul_ijk_naive.c -ftree-vectorize
	gcc -o output/mat_mul_funroll_loops optimization_compiler/mat_mul_ijk_naive.c -funroll-loops
	gcc -o output/mat_mul_o1 optimization_compiler/mat_mul_ijk_naive.c -O1
	gcc -o output/mat_mul_o2 optimization_compiler/mat_mul_ijk_naive.c -O2
	gcc -o output/mat_mul_o3 optimization_compiler/mat_mul_ijk_naive.c -O3
	gcc -o output/mat_mul_naive optimization_compiler/mat_mul_ijk_naive.c

run_opt_compiler:
	./output/mat_mul_naive $(SIZE)
	./output/mat_mul_ftree_vectorize $(SIZE)
	./output/mat_mul_funroll_loops $(SIZE)
	./output/mat_mul_o1 $(SIZE)
	./output/mat_mul_o2 $(SIZE)
	./output/mat_mul_o3 $(SIZE)

build_naive:
	gcc -o output/mat_mul_naive task4_hardware_counters/mat_mul_origin.c

build: build_opt_parallel build_opt_simd build_naive

clean:
	rm -rf output/*
