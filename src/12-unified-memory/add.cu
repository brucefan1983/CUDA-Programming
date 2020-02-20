#include "error.cuh" 
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x, *y, *z;
    CHECK(cudaMallocManaged((void **)&x, M));
    CHECK(cudaMallocManaged((void **)&y, M));
    CHECK(cudaMallocManaged((void **)&z, M));

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(x, y, z);

    CHECK(cudaDeviceSynchronize());
    check(z, N);

    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
    CHECK(cudaFree(z));
    return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

