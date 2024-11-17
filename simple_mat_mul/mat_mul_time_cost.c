#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */

/*
 *  main - program entry point
 *      @argc: number of arguments & program name
 *      @argv: arguments
 */
int32_t
main(int32_t argc, char *argv[])
{
    /* Define array of matrix sizes to test */
    uint32_t sizes[] = {256, 512, 1024, 2048};
    uint32_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    float slow_results[5];
    float fast_results[5];

    /* Run tests for each matrix size */
    for (uint32_t s = 0; s < num_sizes; s++) {
        uint32_t N = sizes[s];

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

        /* perform slow multiplication */
        for (uint32_t i=0; i<N; ++i)
            for (uint32_t j=0; j<N; ++j)
                for (uint32_t k=0; k<N; ++k)
                    r[i*N + j] += m1[i*N + k] * m2[k*N + j];

        /* clock delta */
        t = clock() - t;
        slow_results[s] = ((float)t)/CLOCKS_PER_SEC;

        /* result matrix clear; clock init */
        memset(r, 0, N * N * sizeof(int64_t));
        t = clock();

        /* perform fast(er) multiplication */
        for (uint32_t k=0; k<N; ++k)
            for (uint32_t i=0; i<N; ++i)
                for (uint32_t j=0; j<N; ++j)
                    r[i*N + j] += m1[i*N + k] * m2[k*N + j];

        /* clock delta */
        t = clock() - t;
        fast_results[s] = ((float)t)/CLOCKS_PER_SEC;

        printf("N: %d, Slow: %f, Fast: %f\n", N, slow_results[s], fast_results[s]);

        /* free allocated memory */
        free(m1);
        free(m2);
        free(r);
    }

    /* Print results in the requested format */
    printf("Slow mul: ");
    for (uint32_t i = 0; i < num_sizes; i++) {
        printf("%.2f%s", slow_results[i], i < num_sizes - 1 ? ", " : "\n");
    }

    printf("Fast mul: ");
    for (uint32_t i = 0; i < num_sizes; i++) {
        printf("%.2f%s", fast_results[i], i < num_sizes - 1 ? ", " : "\n");
    }

    return 0;
}
