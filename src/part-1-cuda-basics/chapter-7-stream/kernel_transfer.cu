#include "error.cuh"
#include <stdio.h>
#include <math.h>
#define EPSILON 1.0e-14

void __global__ sum(double *x, double *y, double *z, int N);
void run(int N_streams);

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 3.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

int main(void)
{
    for (int n = 1; n < 128; n *=2)
    {
        run(n);
    }
    return 0;
}

void run(int N_streams)
{
    int N_all = 1 << 20;
    int M_all = sizeof(double) * N_all;
    int N1 = N_all / N_streams;
    int M1 = M_all / N_streams;

    double *x, *y, *z;
    CHECK(cudaMallocHost((void**)&x, M_all));
    CHECK(cudaMallocHost((void**)&y, M_all));
    CHECK(cudaMallocHost((void**)&z, M_all));
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

    cudaStream_t *streams = (cudaStream_t *) 
        malloc(N_streams * sizeof(cudaStream_t));
    for (int i = 0; i < N_streams; i++)
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
        CHECK(cudaMemcpyAsync(g_x + offset, x + offset, M1, 
            cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(g_y + offset, y + offset, M1, 
            cudaMemcpyHostToDevice, streams[i]));

        int block_size = 128;
        int grid_size = (N1 - 1) / block_size + 1;
        sum<<<grid_size, block_size, 0, streams[i]>>>
        (g_x + offset, g_y + offset, g_z + offset, N1);

        CHECK(cudaMemcpyAsync(z + offset, g_z + offset, M1, 
            cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("%d\t%g\n", N_streams, elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    for (int i = 0 ; i < N_streams; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    cudaDeviceSynchronize();
    check(z, N_all);

    CHECK(cudaFreeHost(x))
    CHECK(cudaFreeHost(y))
    CHECK(cudaFreeHost(z))
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
}

void __global__ sum(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        for (int i = 0; i < 40; ++i)
        {
            z[n] = x[n] + y[n];
        }
    }
}
