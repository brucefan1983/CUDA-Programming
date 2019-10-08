#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-14;
void __global__ add1(const double *x, const double *y, double *z, const int N);
void __global__ add2(const double *x, const double *y, double *z, const int N);
void __global__ add3(const double *x, const double *y, double *z, const int N);
void check(const double *z, int N);

int main(void)
{
    const int N = 100000001;
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
    cudaMalloc((void **)&g_x, M);
    cudaMalloc((void **)&g_y, M);
    cudaMalloc((void **)&g_z, M);
    cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;

    add1<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);

    add2<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);

    add3<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);

    free(x);
    free(y);
    free(z);
    cudaFree(g_x);
    cudaFree(g_y);
    cudaFree(g_z);
    return 0;
}

double __device__ add1_device(const double x, const double y)
{
    return (x + y);
}

void __device__ add2_device(const double x, const double y, double *z)
{
    *z = x + y;
}

void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y;
}

void __global__ add1(const double *x, const double *y, double *z, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = add1_device(x[n], y[n]);
    }
}

void __global__ add2(const double *x, const double *y, double *z, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add2_device(x[n], y[n], &z[n]);
    }
}

void __global__ add3(const double *x, const double *y, double *z, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add3_device(x[n], y[n], z[n]);
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

