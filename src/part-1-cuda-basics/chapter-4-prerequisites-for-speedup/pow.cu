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
    cudaMalloc((void **)&g_x, M);
    cudaMalloc((void **)&g_y, M);
    cudaMalloc((void **)&g_z, M);
    cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice);
    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    clock_t time_begin = clock();
    power<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    cudaDeviceSynchronize();
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
    printf("Time used for device function = %f s.\n", time_used);
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    check(z, N);
    free(x); free(y); free(z);
    cudaFree(g_x); cudaFree(g_y); cudaFree(g_z);
    return 0;
}

void __global__ power(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) { z[n] = pow(x[n], y[n]); }
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - 1.0) > EPSILON) { has_error = 1; }
    }
    if (has_error) { printf("Has errors.\n"); }
    else { printf("No errors.\n"); }
}

