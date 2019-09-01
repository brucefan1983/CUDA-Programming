#include "error.cuh"
#include <math.h>
#include <stdio.h>
#define EPSILON 1.0e-14
void __global__ sum(double *x, double *y, double *z, int N);
void run(int N_streams);

int main(void)
{
    for (int n = 0; n < 30; ++n)
    {
        run(n+1);
    }
    return 0;
}

void run(int N_streams)
{
    int N1 = 1000;
    int M1 = sizeof(double) * N1;
    int N_all = N1 * N_streams;
    int M_all = M1 * N_streams;
    double *x = (double*) malloc(M_all);
    double *y = (double*) malloc(M_all);
    double *z = (double*) malloc(M_all);
    for (int n = 0; n < N_all; ++n)
    {
        x[n] = 1.0;
        y[n] = 2.0;
        z[n] = 0.0;
    }
    double *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M_all))
    CHECK(cudaMalloc((void **)&g_y, M_all))
    CHECK(cudaMalloc((void **)&g_z, M_all))
    CHECK(cudaMemcpy(g_x, x, M_all, cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(g_y, y, M_all, cudaMemcpyHostToDevice))

    cudaStream_t *streams = (cudaStream_t *) 
        malloc(N_streams * sizeof(cudaStream_t));
    for (int i = 0 ; i < N_streams ; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    for (int i = 0; i < N_streams; i++)
    {
        int offset = i * N1;
        int block_size = 128;
        int grid_size = (N1 - 1) / block_size + 1;
        sum<<<grid_size, block_size, 0, streams[i]>>>
        (g_x + offset, g_y + offset, g_z + offset, N1);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("%d\t%g\n", N_streams, elapsed_time);

    for (int i = 0 ; i < N_streams; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(z, g_z, M_all, cudaMemcpyDeviceToHost))

    free(x);
    free(y);
    free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
}

void __global__ sum(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        for (int i = 0; i < 1000000; ++i)
        {
            z[n] = x[n] + y[n];
        }
    }
}

