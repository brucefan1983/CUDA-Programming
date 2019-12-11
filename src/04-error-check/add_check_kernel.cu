#include "error.cuh"
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 1280;
    const int grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
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
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

