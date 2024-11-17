#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>
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
    
    memset(r1, 0, N * N * sizeof(int64_t));
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

    memset(r1, 0, N * N * sizeof(int64_t));
    /* 并行矩阵乘法 */
    double parallel_SIMD_start_time = omp_get_wtime();  // 记录并行开始时间
    int64_t  *m2_T = malloc(N * N * sizeof(int64_t));
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            m2_T[j * N + i] = m2[i * N + j];
        }
    }
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            __m512i sum = _mm512_setzero_si512();
            for (uint32_t k = 0; k < N; k += 8) {
                __m512i m1_vec = _mm512_loadu_si512((__m512i*)&m1[i * N + k]);
                __m512i m2_vec = _mm512_loadu_si512((__m512i*)&m2_T[j * N + k]);
                sum = _mm512_add_epi64(sum, _mm512_mullo_epi64(m1_vec, m2_vec));
            }
        int64_t* p1 = (int64_t*)&sum;
        r1[i * N + j] = p1[0] + p1[1] + p1[2] + p1[3] + p1[4] + p1[5] + p1[6] + p1[7];
        }
    }
    double parallel_SIMD_end_time = omp_get_wtime();  // 记录并行结束时间
    double parallel_SIMD_time = parallel_SIMD_end_time - parallel_SIMD_start_time;  // 计算并行时间
    printf("Multiplication2&SIMD completed in: %.6f seconds\n", parallel_SIMD_time);

    /* 释放内存 */
    free(m1);
    free(m2);
    free(r1);

    printf("Memory freed successfully.\n");
    return 0;
}
