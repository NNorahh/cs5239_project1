#include <stdio.h>      
#include <time.h>
#include <inttypes.h>        
#include <stdlib.h>     
#include <string.h>     
#include <stdint.h>     
#include <pthread.h>
#include <immintrin.h>
#include <sys/time.h>
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


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE 32

/* Rename timer_t to my_timer_t to avoid conflict */
typedef struct {
    struct timeval start;
    struct timeval end;
} my_timer_t;

// Structure for parallel execution
typedef struct {
    uint32_t thread_id;
    uint32_t num_threads;
    uint32_t N;
    int64_t *m1;
    int64_t *m2;
    int64_t *r;
} thread_arg_t;

void timer_start(my_timer_t* timer) {
    gettimeofday(&timer->start, NULL);
}

double timer_end(my_timer_t* timer) {
    gettimeofday(&timer->end, NULL);
    return (timer->end.tv_sec - timer->start.tv_sec) + 
           (timer->end.tv_usec - timer->start.tv_usec) / 1000000.0;
}

void* matrix_multiply_parallel_simd(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    uint32_t chunk_size = targ->N / targ->num_threads;
    uint32_t start = targ->thread_id * chunk_size;
    uint32_t end = (targ->thread_id == targ->num_threads - 1) ? targ->N : start + chunk_size;

    for (uint32_t ii = start; ii < end; ii += BLOCK_SIZE) {
        for (uint32_t jj = 0; jj < targ->N; jj += BLOCK_SIZE) {
            for (uint32_t kk = 0; kk < targ->N; kk += BLOCK_SIZE) {
            
                for (uint32_t i = ii; i < MIN(ii + BLOCK_SIZE, end); i++) {
                    for (uint32_t j = jj; j < MIN(jj + BLOCK_SIZE, targ->N); j++) {
                        __m512i sum = _mm512_setzero_si512(); // 初始化SIMD累加器为零
                        for (uint32_t k = kk; k < MIN(kk + BLOCK_SIZE, targ->N); k+=8) {
                            __m512i m1_vec = _mm512_loadu_si512((__m512i*)&targ->m1[i * targ->N + k]);
                            __m512i m2_vec = _mm512_loadu_si512((__m512i*)&targ->m2[j * targ->N + k]);
                            sum = _mm512_add_epi64(sum, _mm512_mullo_epi64(m1_vec, m2_vec));
                        }
                        int64_t temp[8];
                        _mm512_storeu_si512((__m512i*)temp, sum); // 将SIMD结果存储到普通数组
                        targ->r[i * targ->N + j] += temp[0] + temp[1] + temp[2] + temp[3] +
                                                    temp[4] + temp[5] + temp[6] + temp[7];
                    }
                }
            }
        }
    }
    return NULL;
}

// Function to verify results
int verify_results(uint32_t N, int64_t *r1, int64_t *r2) {
    for (uint32_t i = 0; i < N * N; i++) {
        if (r1[i] != r2[i]) {
            return 0;
        }
    }
    return 1;
}

// Function to initialize matrices
void init_matrices(uint32_t N, int64_t *m1, int64_t *m2) {
    for (uint32_t i = 0; i < N * N; i++) {
        m1[i] = i % 100;  // Using modulo to keep numbers manageable
        m2[i] = i % 100;
    }
}

int main(int32_t argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./mat_mul <matrix_size> <num_threads>\n");
        printf("Example: ./mat_mul 1000 4\n");
        return -1;
    }

    memory_stats_t stats_start, stats_current;
    get_memory_stats(&stats_start);

    uint32_t N = atoi(argv[1]);
    uint32_t num_threads = atoi(argv[2]);
    
    if (N < 1 || num_threads < 1) {
        printf("Error: Matrix size and number of threads must be positive\n");
        return -1;
    }

    // Allocate matrices
    int64_t *m1 = malloc(N * N * sizeof(int64_t));
    int64_t *m2 = malloc(N * N * sizeof(int64_t));
    int64_t *r_simple = malloc(N * N * sizeof(int64_t));
    int64_t *r_blocked = malloc(N * N * sizeof(int64_t));
    int64_t *r_parallel = malloc(N * N * sizeof(int64_t));
    int64_t *r_parallel_SIMD = malloc(N * N * sizeof(int64_t));

    if (!m1 || !m2 || !r_simple || !r_blocked || !r_parallel) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -1;
    }

    // Initialize matrices
    init_matrices(N, m1, m2);

    // 4. Parallel blocked multiplication & SIMD
    pthread_t* threads_SIMD = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t* thread_SIMD_args = malloc(num_threads * sizeof(thread_arg_t));

    if (!threads_SIMD || !thread_SIMD_args) {
        fprintf(stderr, "Thread allocation failed!\n");
        return -1;
    }

    memset(r_parallel_SIMD, 0, N * N * sizeof(int64_t));
    my_timer_t timer_seq;
    timer_start(&timer_seq);

    // Transpose matrix m2
    int64_t *m2_T = malloc(N * N * sizeof(int64_t));
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            m2_T[j * N + i] = m2[i * N + j];
        }
    }

    // Create threads
    for (uint32_t i = 0; i < num_threads; i++) {
        thread_SIMD_args[i].thread_id = i;
        thread_SIMD_args[i].num_threads = num_threads;
        thread_SIMD_args[i].N = N;
        thread_SIMD_args[i].m1 = m1;
        thread_SIMD_args[i].m2 = m2_T;
        thread_SIMD_args[i].r = r_parallel_SIMD;
        
        if (pthread_create(&threads_SIMD[i], NULL, matrix_multiply_parallel_simd, &thread_SIMD_args[i])) {
            fprintf(stderr, "Error creating thread %u\n", i);
            return -1;
        }
    }
    // Wait for threads
    for (uint32_t i = 0; i < num_threads; i++) {
        pthread_join(threads_SIMD[i], NULL);
    }
    double time_parallel_SIMD = timer_end(&timer_seq);
    get_memory_stats(&stats_current);


    printf("Parallel Blocked & SIMD Execution Time: %.4f s\n", 
           time_parallel_SIMD);
    print_memory_stats_delta(&stats_start, &stats_current, "Overall");
    printf("---------------------------------------------------\n");

    // Cleanup
    free(m1);
    free(m2);
    free(m2_T);
    free(r_parallel_SIMD);
    free(threads_SIMD);
    free(thread_SIMD_args);

    return 0;
}
