#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))
#define TILE_SIZE 64
#define ROW_SIZE 64


void matrixMultiplyTiled(uint32_t N, int64_t *m1, int64_t *m2, int64_t *r) {
    memset(r, 0, N * N * sizeof(int64_t));

    for (uint32_t k = 0; k < N; k += TILE_SIZE) {
        for (uint32_t i = 0; i < N; i += TILE_SIZE) {
            for (uint32_t j = 0; j < N; j += TILE_SIZE) {

                for (uint32_t k1 = k; k1 < k + TILE_SIZE && k1 < N; ++k1) {
                    for (uint32_t i1 = i; i1 < i + TILE_SIZE && i1 < N; ++i1) {
                        int64_t temp = m1[i1 * N + k1]; 

                        for (uint32_t j1 = j; j1 < j + TILE_SIZE && j1 + 4 <= N; j1 += 4) {
                            r[i1 * N + j1] += temp * m2[k1 * N + j1];
                            r[i1 * N + j1 + 1] += temp * m2[k1 * N + j1 + 1];
                            r[i1 * N + j1 + 2] += temp * m2[k1 * N + j1 + 2];
                            r[i1 * N + j1 + 3] += temp * m2[k1 * N + j1 + 3];
                        }

                        for (uint32_t j1 = j + (TILE_SIZE / 4) * 4; j1 < j + TILE_SIZE && j1 < N; ++j1) {
                            r[i1 * N + j1] += temp * m2[k1 * N + j1];
                        }
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
    printf("\t./mat_mul <N>\n");
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

    /* allocate space for matrices */
    clock_t t;
    uint32_t N   = atoi(argv[1]);
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


    /* perform slow multiplication */
    for (uint32_t i=0; i<N; ++i)             /* line   */
        for (uint32_t j=0; j<N; ++j)         /* column */
            for (uint32_t k=0; k<N; ++k)
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;

    printf("Multiplication 1 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC);


    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();


    /* perform fast(er) multiplication */
    /* change the loop order*/
    for (uint32_t k=0; k<N; ++k)
        for (uint32_t i=0; i<N; ++i)         /* line   */
            for (uint32_t j=0; j<N; ++j)     /* column */
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;

    printf("Multiplication 2 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC); 

    t = clock();
    /* perform fast(er) multiplication by tiling*/
    /* tiling */
    matrixMultiplyTiled(N, m1, m2, r);
    /* clock delta */
    t = clock() - t;
    printf("Multiplication 3 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC); 
    
    return 0;
}