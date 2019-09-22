#include <math.h>
#include <stdio.h>
#define EPSILON 1.0e-14

void __global__ add1(double *x, double *y, double *z, int N);
void __global__ add2(double *x, double *y, double *z, int N);
void __global__ add3(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 100000001;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0; y[n] = 2.0; z[n] = 0.0;
    }

    double *g_x, *g_y, *g_z;
    cudaMalloc((void **)&g_x, M);
    cudaMalloc((void **)&g_y, M);
    cudaMalloc((void **)&g_z, M);
    cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice);

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;

    add1<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);

    add2<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);

    add3<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);

    free(x); free(y); free(z);
    cudaFree(g_x); cudaFree(g_y); cudaFree(g_z);
    return 0;
}

double __device__ add1_device(double x, double y)
{
    return (x + y);
}

void __device__ add2_device(double x, double y, double *z)
{
    *z = x + y;
}

void __device__ add3_device(double x, double y, double &z)
{
    z = x + y;
}

void __global__ add1(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) { z[n] = add1_device(x[n], y[n]); }
}

void __global__ add2(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) { add2_device(x[n], y[n], &z[n]); }
}

void __global__ add3(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) { add3_device(x[n], y[n], z[n]); }
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

