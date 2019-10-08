#include <math.h>
#include <stdio.h>
#include "error.cuh"

const double EPSILON = 1.0e-14;
void __global__ add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
        y[n] = 2.0;
    }

    double *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMalloc((void **)&g_z, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyDeviceToHost))
    CHECK(cudaMemcpy(g_y, y, M, cudaMemcpyDeviceToHost))
    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;
    add<<<grid_size, block_size>>>(g_x, g_y, g_z, N);

    CHECK(cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost))
    check(z, N);

    free(x);
    free(y);
    free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
    return 0;
}

void __global__ add(const double *x, const double *y, double *z, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - 3.0) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

