#include "error.cuh"
#include <math.h>
#include <stdio.h>
#define EPSILON 1.0e-14
void __global__ add(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x, *y, *z;
    CHECK(cudaMallocManaged((void**)&x, M))
    CHECK(cudaMallocManaged((void**)&y, M))
    CHECK(cudaMallocManaged((void**)&z, M))
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0; y[n] = 2.0; z[n] = 0.0;
    }

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    add<<<grid_size, block_size>>>(x, y, z, N);
    
    CHECK(cudaDeviceSynchronize())
    
    check(z, N);

    CHECK(cudaFree(x)) 
    CHECK(cudaFree(y))
    CHECK(cudaFree(z))
    return 0;
}

void __global__ add(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) { z[n] = x[n] + y[n]; }
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
