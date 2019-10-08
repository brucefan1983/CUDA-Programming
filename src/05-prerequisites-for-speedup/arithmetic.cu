#include "error.cuh"
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-14;
#else
    typedef float real;
    const real EPSILON = 1.0e-6;
#endif

void __global__ arithmetic(real *x, const int N);
void check(const real *x, const int N);

int main(int argc, char **argv)
{
    if (argc != 2) 
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    int N = atoi(argv[1]);


    int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *g_x;
    CHECK(cudaMalloc((void **)&g_x, M))

    const int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    arithmetic<<<grid_size, block_size>>>(g_x, N);
   
    cudaMemcpy(x, g_x, M, cudaMemcpyDeviceToHost);
    check(x, N);

    free(x);
    CHECK(cudaFree(g_x))

    return 0;
}

void __global__ arithmetic(real *g_x, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real a = 0;
        for (int m = 0; m < 1000; ++m)
        {
            a++;
        }
        g_x[n] = a;
    }
}

void check(const real *y, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(y[n] - 1000.0) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

