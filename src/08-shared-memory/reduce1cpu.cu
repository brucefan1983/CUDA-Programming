#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}


