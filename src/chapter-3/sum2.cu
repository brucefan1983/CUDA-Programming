#include <math.h> // fabs()
#include <stdio.h>
#define EPSILON 1.0e-14 // a small number
void __global__ sum(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 1024 * 100000;
    int M = sizeof(double) * N;
    double *x = new double[M];
    double *y = new double[M];
    double *z = new double[M];
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
    int grid_size = N / block_size;
    sum<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);
    delete [] x; delete [] y; delete [] z;
    cudaFree(g_x); cudaFree(g_y); cudaFree(g_z);
    return 0;
}

void __global__ sum(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        double diff = fabs(z[n] - 3.0);
        if (diff > EPSILON) { has_error = 1; }
    }
    if (has_error) { printf("Has errors.\n"); }
    else { printf("No errors.\n"); }
}

