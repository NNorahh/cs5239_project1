#include <stdio.h>      
#include <time.h>       
#include <stdlib.h>     
#include <string.h>     
#include <stdint.h>     
#include <pthread.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE 32

// Structure for parallel execution
typedef struct {
    uint32_t thread_id;
    uint32_t num_threads;
    uint32_t N;
    int64_t *m1;
    int64_t *m2;
    int64_t *r;
} thread_arg_t;

// Sequential matrix multiplication without blocking
void matrix_multiply_simple(uint32_t N, int64_t *m1, int64_t *m2, int64_t *r) {
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < N; k++) {
                sum += m1[i * N + k] * m2[k * N + j];
            }
            r[i * N + j] = sum;
        }
    }
}

// Sequential matrix multiplication with blocking
void matrix_multiply_blocked(uint32_t N, int64_t *m1, int64_t *m2, int64_t *r) {
    memset(r, 0, N * N * sizeof(int64_t));
    
    for (uint32_t ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (uint32_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (uint32_t kk = 0; kk < N; kk += BLOCK_SIZE) {
                for (uint32_t i = ii; i < MIN(ii + BLOCK_SIZE, N); i++) {
                    for (uint32_t j = jj; j < MIN(jj + BLOCK_SIZE, N); j++) {
                        int64_t sum = r[i * N + j];
                        for (uint32_t k = kk; k < MIN(kk + BLOCK_SIZE, N); k++) {
                            sum += m1[i * N + k] * m2[k * N + j];
                        }
                        r[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

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

    if (!m1 || !m2 || !r_simple || !r_blocked || !r_parallel) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -1;
    }

    // Initialize matrices
    init_matrices(N, m1, m2);

    // 1. Simple sequential multiplication
    clock_t start_time = clock();
    matrix_multiply_simple(N, m1, m2, r_simple);
    double time_simple = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;

    // 2. Blocked sequential multiplication
    start_time = clock();
    matrix_multiply_blocked(N, m1, m2, r_blocked);
    double time_blocked = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;

    // 3. Parallel blocked multiplication
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t* thread_args = malloc(num_threads * sizeof(thread_arg_t));

    if (!threads || !thread_args) {
        fprintf(stderr, "Thread allocation failed!\n");
        return -1;
    }

    memset(r_parallel, 0, N * N * sizeof(int64_t));
    start_time = clock();

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
    double time_parallel = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;

    // Print performance results
    printf("\nPerformance Results for %ux%u Matrix Multiplication:\n", N, N);
    printf("---------------------------------------------------\n");
    printf("1. Simple Sequential:  %.3f seconds\n", time_simple);
    printf("2. Blocked Sequential: %.3f seconds (%.2fx speedup)\n", 
           time_blocked, time_simple/time_blocked);
    printf("3. Parallel Blocked:   %.3f seconds (%.2fx speedup)\n", 
           time_parallel, time_simple/time_parallel);
    
    // Verify results
    printf("\nVerifying Results:\n");
    printf("Blocked vs Simple: %s\n", 
           verify_results(N, r_simple, r_blocked) ? "MATCH" : "MISMATCH");
    printf("Parallel vs Simple: %s\n", 
           verify_results(N, r_simple, r_parallel) ? "MATCH" : "MISMATCH");

    // Print sample results
    printf("\nSample Results (top-left 3x3 corner):\n");
    for (uint32_t i = 0; i < MIN(3, N); i++) {
        for (uint32_t j = 0; j < MIN(3, N); j++) {
            printf("%lld ", r_simple[i * N + j]);
        }
        printf("\n");
    }

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
