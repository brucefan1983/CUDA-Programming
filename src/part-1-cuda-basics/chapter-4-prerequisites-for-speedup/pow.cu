#include "error.cuh"
#include <math.h> // fabs()
#include <stdio.h>
#include <time.h> // clock(), clock_t, and CLOCKS_PER_SEC
#define EPSILON 1.0e-14 // a small number
void __global__ power(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n) { x[n] = 1.0; y[n] = 2.0; }
    double *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMalloc((void **)&g_z, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice))

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    power<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
   
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);
    free(x);
    free(y);
    free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
    return 0;
}

void __global__ power(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = pow(x[n], y[n]);
    }
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 1.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

