#include "error.cuh"
#include <math.h>
#include <stdio.h>
#include <time.h>
#define EPSILON 1.0e-14
void __global__ arithmetic(double *x, double *y, int N);
void check(double *z, int N);

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 2.0; y[n] = 0.0;
    }
    double *g_x, *g_y;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice))

    int grid_size = (N - 1) / block_size + 1;
    arithmetic<<<grid_size, block_size>>>(g_x, g_y, N);
   
    cudaMemcpy(y, g_yy, M, cudaMemcpyDeviceToHost);
    check(, N);

    free(x); free(y); free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
    return 0;
}

void __global__ arithmetic(double *x, double *y, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) 
    {
        double x1 = x[n];
        double x30 = pow(x1, 30.0);
        double sin_x = sin(x30);
        double cos_x = cos(x30);
        y[n] = sin_x * sin_x + cos_x * cos_x;
    }
}

void check(double *y, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(y[n] - 1.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

