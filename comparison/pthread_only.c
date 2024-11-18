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

void timer_start(my_timer_t* timer) {
    gettimeofday(&timer->start, NULL);
}

double timer_end(my_timer_t* timer) {
    gettimeofday(&timer->end, NULL);
    return (timer->end.tv_sec - timer->start.tv_sec) + 
           (timer->end.tv_usec - timer->start.tv_usec) / 1000000.0;
}

// Structure for parallel execution
typedef struct {
    uint32_t thread_id;
    uint32_t num_threads;
    uint32_t N;
    int64_t *m1;
    int64_t *m2;
    int64_t *r;
} thread_arg_t;

// Parallel matrix multiplication (thread function)
void* matrix_multiply_parallel(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    uint32_t chunk_size = targ->N / targ->num_threads;
    uint32_t start = targ->thread_id * chunk_size;
    uint32_t end = (targ->thread_id == targ->num_threads - 1) ? targ->N : start + chunk_size;

    for (uint32_t ii = start; ii < end; ii += BLOCK_SIZE) {
        for (uint32_t jj = 0; jj < targ->N; jj += BLOCK_SIZE) {
            for (uint32_t kk = 0; kk < targ->N; kk += BLOCK_SIZE) {
                for (uint32_t i = ii; i < MIN(ii + BLOCK_SIZE, end); i++) {
                    for (uint32_t j = jj; j < MIN(jj + BLOCK_SIZE, targ->N); j++) {
                        int64_t sum = 0;
                        for (uint32_t k = kk; k < MIN(kk + BLOCK_SIZE, targ->N); k++) {
                            sum += targ->m1[i * targ->N + k] * targ->m2[k * targ->N + j];
                        }
                        targ->r[i * targ->N + j] += sum;
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

    // 3. Parallel blocked multiplication
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t* thread_args = malloc(num_threads * sizeof(thread_arg_t));

    if (!threads || !thread_args) {
        fprintf(stderr, "Thread allocation failed!\n");
        return -1;
    }

    memset(r_parallel, 0, N * N * sizeof(int64_t));
    my_timer_t timer_seq;
    timer_start(&timer_seq);

    // Create threads
    for (uint32_t i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].N = N;
        thread_args[i].m1 = m1;
        thread_args[i].m2 = m2;
        thread_args[i].r = r_parallel;
        
        if (pthread_create(&threads[i], NULL, matrix_multiply_parallel, &thread_args[i])) {
            fprintf(stderr, "Error creating thread %u\n", i);
            return -1;
        }
    }

    // Wait for threads
    for (uint32_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    double time_parallel = timer_end(&timer_seq);
    get_memory_stats(&stats_current);

    printf("Parallel Blocked (only) Execution Time: %.4f s\n", 
           time_parallel);
    print_memory_stats_delta(&stats_start, &stats_current, "Overall");
    printf("---------------------------------------------------\n");

    // Cleanup
    free(m1);
    free(m2);
    free(r_simple);
    free(r_blocked);
    free(r_parallel);
    free(threads);
    free(thread_args);

    return 0;
}
