#include <stdlib.h>
#include <stdio.h>
double get_length(double *x, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
    }
    double length = get_length(x, N);
    printf("length = %g.\n", length);
    free(x);
    return 0;
}

void __global__ get_length
(double *g_x, double *g_length, int N, int number_of_patches)
{
    int tid = threadIdx.x;
    g_length[tid] = 0.0;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024;
        if (n < N)
        {
            double x_n = g_x[n];
            g_length[tid] += x_n * x_n;
        }
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            if (tid + offset < N)
            {
                g_length[tid] += g_length[tid + offset];
            }
        }
        __syncthreads();
    }
}

double get_length(double *x, int N)
{
    double *g_length;
    cudaMalloc((void**)&g_length, sizeof(double) * 1024);
    double *g_x;
    cudaMalloc((void**)&g_x, sizeof(double) * N);
    cudaMemcpy(g_x, x, sizeof(double) * N, 
        cudaMemcpyHostToDevice);

    int number_of_patches = (N - 1) / 1024 + 1;
    get_length<<<1, 1024>>>
    (g_x, g_length, N, number_of_patches);

    double *cpu_length = (double *) malloc(sizeof(double));
    cudaMemcpy(cpu_length, g_length, sizeof(double), 
        cudaMemcpyDeviceToHost);
    cudaFree(g_length);
    cudaFree(g_x);
    double length = sqrt(cpu_length[0]);
    free(cpu_length);
    return length;
}

