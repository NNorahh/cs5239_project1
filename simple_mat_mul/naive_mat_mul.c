#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */

int32_t
usage(void)
{
    printf("\t./mat_mul <N> <order>\n");
    printf("\torder: 'ijk' or 'kij'\n");
    return -1;
}

int32_t
main(int32_t argc, char *argv[])
{
    if (argc != 3)
            return usage();

    /* allocate space for matrices */
    clock_t t;
    uint32_t N   = atoi(argv[1]);
    char *order  = argv[2];
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

    /* perform multiplication based on order */
    if (strcmp(order, "ijk") == 0) {
        for (uint32_t i=0; i<N; ++i)
            for (uint32_t j=0; j<N; ++j)
                for (uint32_t k=0; k<N; ++k)
                    r[i*N + j] += m1[i*N + k] * m2[k*N + j];
    } else if (strcmp(order, "kij") == 0) {
        for (uint32_t k=0; k<N; ++k)
            for (uint32_t i=0; i<N; ++i)
                for (uint32_t j=0; j<N; ++j)
                    r[i*N + j] += m1[i*N + k] * m2[k*N + j];
    } else {
        printf("Invalid order. Use 'ijk' or 'kij'\n");
        return -1;
    }

    /* clock delta */
    t = clock() - t;

    printf("Multiplication 1 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC);

    free(m1);
    free(m2);
    free(r);
    return 0;
}
