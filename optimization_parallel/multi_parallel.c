#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

int32_t usage(void) {
    printf("Usage: ./mat_mul <N> <num_threads>\n");
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc != 3)
        return usage();

    /* 解析并验证输入参数 */
    uint32_t N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (N == 0 || num_threads <= 0) {
        fprintf(stderr, "Error: N and num_threads must be positive integers.\n");
        return -1;
    }

    omp_set_num_threads(num_threads);

    /* 分配矩阵内存空间 */
    int64_t *m1 = (int64_t *)malloc(N * N * sizeof(int64_t));
    int64_t *m2 = (int64_t *)malloc(N * N * sizeof(int64_t));
    int64_t *r1 = (int64_t *)malloc(N * N * sizeof(int64_t));

    if (!m1 || !m2 || !r1) {
        fprintf(stderr, "Memory allocation failed!\n");
        free(m1);
        free(m2);
        free(r1);
        return -1;
    }
    printf("Memory allocated successfully for %u x %u matrices.\n", N, N);

    /* 初始化矩阵 */
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            m1[i * N + j] = i * N + j;
            m2[i * N + j] = i * N + j;
        }
    }
    printf("Matrices initialized.\n");

    /* 传统算法（单线程）矩阵乘法 */
    double start_time = omp_get_wtime();  // 记录开始时间
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < N; k++) {
                sum += m1[i * N + k] * m2[k * N + j];
            }
            r1[i * N + j] = sum;
        }
    }
    double end_time = omp_get_wtime();  // 记录结束时间
    double single_thread_time = end_time - start_time;  // 计算单线程时间
    printf("Multiplication completed in: %.6f seconds\n", single_thread_time);

    /* 并行矩阵乘法 */
    double parallel_start_time = omp_get_wtime();  // 记录并行开始时间
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < N; k++) {
                sum += m1[i * N + k] * m2[k * N + j];
            }
            r1[i * N + j] = sum;
        }
    }
    double parallel_end_time = omp_get_wtime();  // 记录并行结束时间
    double parallel_time = parallel_end_time - parallel_start_time;  // 计算并行时间
    printf("Multiplication2 completed in: %.6f seconds\n", parallel_time);

    /* 释放内存 */
    free(m1);
    free(m2);
    free(r1);

    printf("Memory freed successfully.\n");
    return 0;
}
