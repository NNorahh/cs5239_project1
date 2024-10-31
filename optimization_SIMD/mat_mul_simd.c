#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <immintrin.h>  /* AVX2                           */

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))

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

    /* TODO: count L2 cache misses for the next block using RDPMC */

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

    /* TODO: count L2 cache misses for the next block using RDPMC */

    /* perform fast(er) multiplication */
    for (uint32_t k=0; k<N; ++k)
        for (uint32_t i=0; i<N; ++i)         /* line   */
            for (uint32_t j=0; j<N; ++j)     /* column */
                r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    /* clock delta */
    t = clock() - t;

    printf("Multiplication 2 finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC); 
    /* result matrix clear; clock init */

    // memset(r, 0, N * N * sizeof(int64_t));
    // t = clock();

    // /* TODO: count L2 cache misses for the next block using RDPMC */

    // /* perform fast(er) multiplication */
    // for (uint32_t i=0; i<N; ++i)
    //     for (uint32_t k=0; k<N; ++k)         /* line   */
    //         for (uint32_t j=0; j<N; ++j)     /* column */
    //             r[i*N + j] += m1[i*N + k] * m2[k*N + j];

    // /* clock delta */
    // t = clock() - t;

    // printf("Multiplication 3 finished in %6.2f s\n",
    //        ((float)t)/CLOCKS_PER_SEC); 

    /* result matrix clear; clock init */

    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; j += 4) {
            __m256i sum = _mm256_setzero_si256();
            for (uint32_t k = 0; k < N; ++k) {
                __m256i m1_vec = _mm256_set1_epi64x(m1[i * N + k]);
                __m256i m2_vec = _mm256_loadu_si256((__m256i*)&m2[k * N + j]);
                sum = _mm256_add_epi64(sum, _mm256_mul_epu32(m1_vec, m2_vec));
            }
            _mm256_storeu_si256((__m256i*)&r[i * N + j], sum);
        }
    }
    t = clock() - t;
    printf("SIMD Multiplication 1 finished in %6.2f s\n",
        ((float)t) / CLOCKS_PER_SEC);
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t k = 0; k < N; ++k) {
            __m256i m1_vec = _mm256_set1_epi64x(m1[i * N + k]);
            for (uint32_t j = 0; j < N; j += 4) {
                __m256i m2_vec = _mm256_loadu_si256((__m256i*)&m2[k * N + j]);
                __m256i sum = _mm256_loadu_si256((__m256i*)&r[i * N + j]);
                sum = _mm256_add_epi64(sum, _mm256_mul_epu32(m1_vec, m2_vec));
                _mm256_storeu_si256((__m256i*)&r[i * N + j], sum);
            }
        }
    }
    t = clock() - t;
    printf("SIMD Multiplication 2 finished in %6.2f s\n",
        ((float)t) / CLOCKS_PER_SEC);

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();
    int block = 32;
    for (uint32_t i = 0; i < N; i += block) {
        for (uint32_t j = 0; j < N; j += block) {
            for (uint32_t i1 = i; i1 < i + block && i1 < N; i1++) {
                for (uint32_t j1 = j; j1 < j + block && j1 < N; j1++) {
                    __m512i res_vec = _mm512_setzero_si512();
                    for (uint32_t k1 = 0; k1 < N; k1 += 16) {
                        __m512i m1_vec = _mm512_loadu_si512((__m512i*)&m1[i1 * N + k1]);
                        __m512i m2_vec = _mm512_loadu_si512((__m512i*)&m2[j1 * N + k1]);
                        res_vec = _mm512_add_epi32(res_vec, _mm512_mullo_epi32(m1_vec, m2_vec));
                    }
                    int* p1 = (int*)&res_vec;
                    r[i1 * N + j1] += (p1[0] + p1[1] + p1[2] + p1[3] + p1[4] + p1[5] + p1[6] + p1[7] +
                                       p1[8] + p1[9] + p1[10] + p1[11] + p1[12] + p1[13] + p1[14] + p1[15]);
                }
            }
        }
    }
    t = clock() - t;
    printf("SIMD & Tiling(32) Multiplication finished in %6.2f s\n",
        ((float)t) / CLOCKS_PER_SEC);

    return 0;
}
