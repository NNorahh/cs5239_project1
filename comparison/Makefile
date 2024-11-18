# 指定编译器和通用编译选项
CC = gcc
CFLAGS = -mavx512f -mavx512dq

# OpenMP 编译选项
OMPFLAGS = -fopenmp

# pthread 链接选项
PTHREAD_LIB = -lpthread
N = 1024           # 矩阵大小
NUM_THREADS = 20    # 线程数量
# 定义每个可执行文件的目标
all: baseline openmp_simd openmp_tiling openmp_only openmp_simd_tiling pthread_simd pthread_only simd_only openmp_simd_ikj run

# 编译 baseline
baseline: baseline.c
	$(CC) baseline.c -o baseline

ikj: ikj.c
	$(CC) ikj.c -o ikj

tiling: tiling.c
	$(CC) tiling.c -o tiling

# 编译 openmp_simd
openmp_simd: openmp_simd.c
	$(CC) $(CFLAGS) $(OMPFLAGS) openmp_simd.c -o openmp_simd

# 编译 openmp
openmp_only: openmp_only.c
	$(CC) $(CFLAGS) $(OMPFLAGS) openmp_only.c -o openmp_only

# 编译 pthread_only
pthread_only: pthread_only.c
	$(CC) $(CFLAGS) $(PTHREAD_LIB) pthread_only.c -o pthread_only

# 编译 pthread_simd
pthread_simd: pthread_simd.c
	$(CC) $(CFLAGS) $(PTHREAD_LIB) pthread_simd.c -o pthread_simd

# 编译 simd_only
simd_only: simd_only.c
	$(CC) $(CFLAGS) simd_only.c -o simd_only

# 运行所有目标文件
PERF_EVENTS = cycles,page-faults,instructions,cache-references,cache-misses,context-switches,mem-loads,mem-stores
OUTPUT_FILE = perf_results.txt

run: baseline openmp_simd openmp_only pthread_only pthread_simd simd_only ikj tiling
	@> $(OUTPUT_FILE)  # 清空文件内容

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running baseline..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./baseline $(N) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running ikj..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./ikj $(N) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running tiling_only..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./tiling $(N) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running pthread_only..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./pthread_only $(N) $(NUM_THREADS) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running openmp_only..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./openmp_only $(N) $(NUM_THREADS) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running simd_only..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./simd_only $(N) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running pthread_simd..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./pthread_simd $(N) $(NUM_THREADS) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

	@echo "*****************************" | tee -a $(OUTPUT_FILE)
	@echo "Running openmp_simd..." | tee -a $(OUTPUT_FILE)
	@perf stat -e $(PERF_EVENTS) ./openmp_simd $(N) $(NUM_THREADS) >> $(OUTPUT_FILE) 2>&1
	@echo "*****************************" | tee -a $(OUTPUT_FILE)

# 清理所有生成的文件
clean:
	rm -f openmp_simd openmp_only pthread_only pthread_simd simd_only ikj tiling
