#include <math.h>
#include <stdio.h>
#include "error.cuh"

#ifdef USE_DP
    typedef double real;
    #define EPSILON 1.0e-14
#else
    typedef float real;
    #define EPSILON 1.0e-6
#endif
void __global__ sum(real *x, real *y, real *z, int N);
void check(real *z, int N);

int main(int argc, char **argv)
{
    if (argc != 1) 
    { printf("requires 1 argument\n"); exit(1); }
    int num_of_repeats = atoi(argv[1]);

    int N = 100000000;
    int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *y = (real*) malloc(M);
    real *z = (real*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0; y[n] = 2.0; z[n] = 0.0;
    }
    real *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMalloc((void **)&g_z, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice))

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    for (int n = 0; n < num_of_repeats; ++n)
    {
        sum<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    }

    CHECK(cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost))
    check(z, N);

    free(x); free(y); free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
    return 0;
}

void __global__ sum(real *x, real *y, real *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) { z[n] = x[n] + y[n]; }
}

void check(real *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 3.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

