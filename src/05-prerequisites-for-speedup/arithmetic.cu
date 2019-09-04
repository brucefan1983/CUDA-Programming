#include "error.cuh"
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef USE_DP
    typedef double real;
    #define EPSILON 1.0e-14
#else
    typedef float real;
    #define EPSILON 1.0e-6
#endif
void __global__ arithmetic(real *x, int N);
void check(real *x, int N);

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *g_x;
    CHECK(cudaMalloc((void **)&g_x, M))

    int grid_size = (N - 1) / block_size + 1;
    arithmetic<<<grid_size, block_size>>>(g_x, N);
   
    cudaMemcpy(x, g_x, M, cudaMemcpyDeviceToHost);
    check(x, N);

    free(x);
    CHECK(cudaFree(g_x))

    return 0;
}

void __global__ arithmetic(real *g_x, int N)
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

void check(real *x, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(x[n] - 1000.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

