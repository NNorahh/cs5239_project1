#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <pthread.h>    /* pthreads                       */
#include <sys/time.h>   /* getrusage                      */
#include <sys/resource.h> /* getrusage                    */
#include <unistd.h>     /* sysconf                        */

/* Global variables for matrix dimensions and data */
uint32_t N;
int64_t *m1, *m2, *r;

/* Structure for passing thread data */
typedef struct {
    uint32_t start_row;
    uint32_t end_row;
} thread_data_t;

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

/*
 * usage - how to run the program
 */
int32_t usage(void) {
    printf("\t./mat_mul <N>\n");
    return -1;
}

/*
 * Print first few elements of matrix for verification
 */
void print_matrix_sample(uint32_t N, long *m, const char *label) {
    printf("\n%s (showing first 3x3 elements):\n", label);
    for (uint32_t i = 0; i < (N < 3 ? N : 3); ++i) {
        for (uint32_t j = 0; j < (N < 3 ? N : 3); ++j) {
            printf("%3ld ", m[i*N + j]);
        }
        printf("\n");
    }
}

/* Thread function for parallel matrix multiplication */
void *matrix_multiply_parallel(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    uint32_t start = data->start_row;
    uint32_t end = data->end_row;

    for (uint32_t i = start; i < end; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < N; ++k) {
                sum += m1[i * N + k] * m2[k * N + j];
            }
            r[i * N + j] = sum;
        }

    pthread_exit(NULL);
}

int32_t main(int32_t argc, char *argv[]) {
    if (argc != 2)
        return usage();

    memory_stats_t stats_start, stats_current;
    clock_t t_total = clock();
    
    /* Get initial memory statistics */
    get_memory_stats(&stats_start);

    /* Parse input and calculate sizes */
    N = atoi(argv[1]);
    size_t matrix_size = N * N * sizeof(int64_t);
    double matrix_size_mb = matrix_size / (1024.0 * 1024.0);
    
    printf("Matrix size: %u x %u\n", N, N);
    printf("Memory required per matrix: %.2f MB\n", matrix_size_mb);
    printf("Total memory required for 3 matrices: %.2f MB\n", 3 * matrix_size_mb);
    
    /* Allocate matrices */
    m1 = malloc(matrix_size);
    m2 = malloc(matrix_size);
    r = malloc(matrix_size);
    
    if (!m1 || !m2 || !r) {
        printf("Memory allocation failed!\n");
        if (m1) free(m1);
        if (m2) free(m2);
        if (r) free(r);
        return -1;
    }

    /* Initialize matrices */
    printf("\nInitializing matrices...\n");
    for (uint32_t i = 0; i < N * N; ++i) {
        m1[i] = i % 100;  // Using modulo to keep numbers manageable
        m2[i] = i % 100;
    }

    /* Sequential multiplication */
    printf("\nPerforming sequential multiplication...\n");
    memory_stats_t stats_seq_start, stats_seq_end;
    get_memory_stats(&stats_seq_start);
    
    memset(r, 0, matrix_size);
    clock_t t_seq = clock();

    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < N; ++k) {
                sum += m1[i * N + k] * m2[k * N + j];
            }
            r[i * N + j] = sum;
        }

    t_seq = clock() - t_seq;
    get_memory_stats(&stats_seq_end);
    
    printf("\n=== Sequential Multiplication Results ===\n");
    printf("Time: %6.2f seconds\n", ((float)t_seq)/CLOCKS_PER_SEC);
    print_memory_stats_delta(&stats_seq_start, &stats_seq_end, "Sequential");
    print_matrix_sample(N, r, "Sequential Result");

    /* Parallel multiplication */
    printf("\nPerforming parallel multiplication...\n");
    memory_stats_t stats_par_start, stats_par_end;
    get_memory_stats(&stats_par_start);
    
    memset(r, 0, matrix_size);
    clock_t t_par = clock();

    uint32_t num_threads = 4;
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    uint32_t rows_per_thread = N / num_threads;
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? N : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, matrix_multiply_parallel, &thread_data[i]);
    }

    for (uint32_t i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    t_par = clock() - t_par;
    get_memory_stats(&stats_par_end);

    printf("\n=== Parallel Multiplication Results ===\n");
    printf("Time: %6.2f seconds\n", ((float)t_par)/CLOCKS_PER_SEC);
    print_memory_stats_delta(&stats_par_start, &stats_par_end, "Parallel");
    print_matrix_sample(N, r, "Parallel Result");

    /* Overall statistics */
    get_memory_stats(&stats_current);
    t_total = clock() - t_total;
    
    printf("\n=== Overall Program Statistics ===\n");
    printf("Total execution time: %6.2f seconds\n", ((float)t_total)/CLOCKS_PER_SEC);
    print_memory_stats_delta(&stats_start, &stats_current, "Overall");

    /* Clean up */
    free(m1);
    free(m2);
    free(r);

    return 0;
}
