#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */

#define TILE_SIZE 64
#define ROW_SIZE 64

void Tiling(int* r, int* m1, int* m2, int n) {
    int i, j, k, i1, j1, k1;
    for (i = 0; i < n; i += TILE_SIZE)
        for (j = 0; j < n; j += TILE_SIZE)
            for (k = 0; k < n; k += TILE_SIZE)
                /* TILE_SIZE x TILE_SIZE mini matrix multiplications */
                for (i1 = i; i1 < i + TILE_SIZE; i1++)
                    for (j1 = j; j1 < j + TILE_SIZE; j1+=16)
                        for (k1 = k; k1 < k + TILE_SIZE; k1++)
                            r[i1 * n + j1] += m1[i1 * n + k1] * m2[k1 * n + j1];
}


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
    if (argc != 3)
        return usage();

    /* parse N and algorithm choice */
    uint32_t N   = atoi(argv[1]);
    int algorithm = atoi(argv[2]);

    /* allocate space for matrices */
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

    /* perform multiplication based on selected algorithm */
    if (algorithm == 1) {
        // Original IJK multiplication
        for (uint32_t i = 0; i < N; ++i)
            for (uint32_t j = 0; j < N; ++j)
                for (uint32_t k = 0; k < N; ++k)
                    r[i*N + j] += m1[i*N + k] * m2[k*N + j];
    }
    else if (algorithm == 2) {
        // Optimized IKJ multiplication
        for (uint32_t k = 0; k < N; ++k)
            for (uint32_t i = 0; i < N; ++i)
                for (uint32_t j = 0; j < N; ++j)
                    r[i*N + j] += m1[i*N + k] * m2[k*N + j];
    }
    else if (algorithm == 3) {
        // Tiling multiplication
        matrixMultiplyTiled(N, m1, m2, r);
    } else {
        return usage(); // invalid algorithm choice
    }

    /* clock delta */
    t = clock() - t;
    printf("Multiplication finished in %6.2f s\n", ((float)t)/CLOCKS_PER_SEC); 
    return 0;
}
