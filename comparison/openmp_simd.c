#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>
#include <sys/resource.h> 

/* Structure for memory statistics */
typedef struct {
    long page_faults;
    long page_reclaims;
    long peak_memory;  // Peak resident set size
} memory_stats_t;
/*
 * Get memory statistics
 */
void get_memory_stats(memory_stats_t *stats) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        stats->page_faults = usage.ru_majflt;    // Major page faults
        stats->page_reclaims = usage.ru_minflt;  // Minor page faults
        stats->peak_memory = usage.ru_maxrss;    // Peak resident set size in kilobytes
    } else {
        perror("getrusage failed");
        stats->page_faults = 0;
        stats->page_reclaims = 0;
        stats->peak_memory = 0;
    }
}
/*
 * Print memory statistics delta
 */
void print_memory_stats_delta(memory_stats_t *before, memory_stats_t *after, const char *label) {
    printf("\n=== %s Memory Statistics ===\n", label);
    printf("  Major Page Faults: %ld\n", after->page_faults - before->page_faults);
    printf("  Minor Page Faults: %ld\n", after->page_reclaims - before->page_reclaims);
    printf("  Current Peak Memory: %ld KB\n", after->peak_memory);
}


int32_t usage(void) {
    printf("Usage: ./mat_mul <N> <num_threads>\n");
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc != 3)
        return usage();

    memory_stats_t stats_start, stats_current;
    get_memory_stats(&stats_start);

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

    /* 初始化矩阵 */
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            m1[i * N + j] = i * N + j;
            m2[i * N + j] = i * N + j;
        }
    }

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
    double parallel_SIMD_time = omp_get_wtime() - parallel_SIMD_start_time;
    get_memory_stats(&stats_current);

    printf("OpenMp & SIMD Execution Time: %6.4f s\n", parallel_SIMD_time);
    print_memory_stats_delta(&stats_start, &stats_current, "Overall");
    printf("---------------------------------------------------\n");

    /* 释放内存 */
    free(m1);
    free(m2);
    free(r1);

    printf("Memory freed successfully.\n");
    return 0;
}
