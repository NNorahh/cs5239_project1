#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
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

#define TILE_SIZE 64

void matrixMultiplyTiled(uint32_t n, int64_t *m1, int64_t *m2, int64_t *r) {
    int i, j, k, i1, j1, k1;
    // Iterate over blocks in the matrices
    for (i = 0; i < n; i += TILE_SIZE) {
        for (j = 0; j < n; j += TILE_SIZE) {
            for (k = 0; k < n; k += TILE_SIZE) {
                // Perform multiplication within each tile using IJK order
                for (i1 = i; i1 < i + TILE_SIZE && i1 < n; i1++) {
                    for (j1 = j; j1 < j + TILE_SIZE && j1 < n; j1++) {
                        int64_t sum = 0;  // Accumulate the product sum for the current (i1, j1)
                        for (k1 = k; k1 < k + TILE_SIZE && k1 < n; k1++) {
                            sum += m1[i1 * n + k1] * m2[k1 * n + j1];
                        }
                        r[i1 * n + j1] += sum;
                    }
                }
            }
        }
    }
}

/*
 *  usage - how to run the program
 *      @return: -1
 */
int32_t
usage(void)
{
    printf("\t./mat_mul <N> <algorithm>\n");
    printf("\talgorithm: 1 = Original IJK, 2 = IKJ, 3 = Tiling\n");
    return -1;
}

/*
 *  print_matrix - if you need convincing that it works just fine
 *      @N: square matrix size
 *      @m: pointer to matrix
 */
void
print_matrix(uint32_t N, long *m)
{
    for (uint32_t i=0; i<N; ++i) {
        for (uint32_t j=0; j<N; ++j)
            printf("%3ld ", m[i*N + j]);
        printf("\n");
    }
}

/*
 *  main - program entry point
 *      @argc: number of arguments & program name
 *      @argv: arguments
 */
int32_t
main(int32_t argc, char *argv[])
{
    if (argc != 2)
        return usage();

    memory_stats_t stats_start, stats_current;
    get_memory_stats(&stats_start);

    /* allocate space for matrices */
    uint32_t N   = atoi(argv[1]);
    clock_t t;
    int64_t  *m1 = malloc(N * N * sizeof(int64_t));
    int64_t  *m2 = malloc(N * N * sizeof(int64_t));
    int64_t  *r  = malloc(N * N * sizeof(int64_t));

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();
    int i, j, k, i1, j1, k1;
    // Iterate over blocks in the matrices
    for (i = 0; i < N; i += TILE_SIZE) {
        for (j = 0; j < N; j += TILE_SIZE) {
            for (k = 0; k < N; k += TILE_SIZE) {
                // Perform multiplication within each tile using IJK order
                for (i1 = i; i1 < i + TILE_SIZE && i1 < N; i1++) {
                    for (j1 = j; j1 < j + TILE_SIZE && j1 < N; j1++) {
                        int64_t sum = 0;  // Accumulate the product sum for the current (i1, j1)
                        for (k1 = k; k1 < k + TILE_SIZE && k1 < N; k1++) {
                            sum += m1[i1 * N + k1] * m2[k1 * N + j1];
                        }
                        r[i1 * N + j1] += sum;
                    }
                }
            }
        }
    }
    t = clock() - t;
    get_memory_stats(&stats_current);
    printf("Tiling Execution Time: %.4f s\n",
        ((float)t) / CLOCKS_PER_SEC);
    print_memory_stats_delta(&stats_start, &stats_current, "Overall");
    printf("---------------------------------------------------\n");
    return 0;
}
