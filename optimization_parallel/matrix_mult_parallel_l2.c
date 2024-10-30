#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <pthread.h>    /* pthreads                       */

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))

/* Global variables for matrix dimensions and data */
uint32_t N;
int64_t *m1, *m2, *r;

/* Structure for passing thread data */
typedef struct {
    uint32_t start_row;
    uint32_t end_row;
} thread_data_t;

/*
 * usage - how to run the program
 * @return: -1
 */
int32_t usage(void) {
    printf("\t./mat_mul <N>\n");
    return -1;
}

/*
 * print_matrix - if you need convincing that it works just fine
 * @N: square matrix size
 * @m: pointer to matrix
 */
void print_matrix(uint32_t N, long *m) {
    for (uint32_t i=0; i<N; ++i) {
        for (uint32_t j=0; j<N; ++j)
            printf("%3ld ", m[i*N + j]);
        printf("\n");
    }
}

uint64_t read_pmc(void) {
    uint32_t eax, edx;
    // Using counter 0 for L2 cache misses
    rdpmc(0, eax, edx);
    // Combine the low-order 32 bits EAX and high-order 32 bits EDX
    return ((uint64_t)edx << 32) | eax;
}

/* Thread function for parallel matrix multiplication */
void *matrix_multiply_parallel(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    uint32_t start = data->start_row;
    uint32_t end = data->end_row;

    for (uint32_t i = start; i < end; ++i)         /* line   */
        for (uint32_t j = 0; j < N; ++j) {         /* column */
            int64_t sum = 0;
            for (uint32_t k = 0; k < N; ++k) {
                sum += m1[i * N + k] * m2[k * N + j];
            }
            r[i * N + j] = sum;
        }

    pthread_exit(NULL);
}

/*
 * main - program entry point
 * @argc: number of arguments & program name
 * @argv: arguments
 */
int32_t main(int32_t argc, char *argv[]) {
    if (argc != 2)
        return usage();

    /* allocate space for matrices */
    clock_t t;
    N = atoi(argv[1]);
    m1 = malloc(N * N * sizeof(int64_t));
    m2 = malloc(N * N * sizeof(int64_t));
    r = malloc(N * N * sizeof(int64_t));

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    uint64_t l2_miss_count_start = read_pmc();

    /* perform slow multiplication */
    for (uint32_t i=0; i<N; ++i)             /* line   */
        for (uint32_t j=0; j<N; ++j)         /* column */
            for (uint32_t k=0; k<N; ++k)
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;
    printf("Multiplication 1 finished in %6.2f s\n", ((float)t)/CLOCKS_PER_SEC);

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */
    uint64_t l2_miss_count_end = read_pmc();
    uint64_t l2_miss_count = l2_miss_count_end - l2_miss_count_start;
    printf("L2 cache misses: %lu\n", l2_miss_count);

    // Record the L2 cache misses for the next block
    l2_miss_count_start = read_pmc();

    /* perform parallel multiplication */
    uint32_t num_threads = 4;  // Set the number of threads
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    uint32_t rows_per_thread = N / num_threads;
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? N : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, matrix_multiply_parallel, &thread_data[i]);
    }

    /* Wait for all threads to finish */
    for (uint32_t i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    /* clock delta */
    t = clock() - t;
    printf("Parallel Multiplication 2 finished in %6.2f s\n", ((float)t)/CLOCKS_PER_SEC);

    // Count L2 cache misses
    l2_miss_count_end = read_pmc();
    l2_miss_count = l2_miss_count_end - l2_miss_count_start;
    printf("L2 cache misses: %lu\n", l2_miss_count);

    /* free allocated memory */
    free(m1);
    free(m2);
    free(r);

    return 0;
}

