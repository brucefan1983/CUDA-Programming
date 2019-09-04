#include <math.h>
#include <stdio.h>
#include "error.cuh"
#define EPSILON 1.0e-14
void __global__ add(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 1024 * 100000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0; y[n] = 2.0; z[n] = 0.0;
    }

    double *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMalloc((void **)&g_z, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyDeviceToHost))
    CHECK(cudaMemcpy(g_y, y, M, cudaMemcpyDeviceToHost))
    int block_size = 128;
    int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(g_x, g_y, g_z, N);

    CHECK(cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost))
    check(z, N);

    free(x); free(y); free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
    return 0;
}

void __global__ add(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 3.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

