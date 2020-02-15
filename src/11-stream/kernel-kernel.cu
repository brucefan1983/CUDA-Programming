#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N1 = 1024;
const int MAX_NUM_STREAMS = 30;
const int N = N1 * MAX_NUM_STREAMS;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N1 - 1) / block_size + 1;
cudaStream_t streams[MAX_NUM_STREAMS];

void timing(const real *d_x, const real *d_y, real *d_z, const int num);

int main(void)
{
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    for (int n = 0 ; n < MAX_NUM_STREAMS; ++n)
    {
        CHECK(cudaStreamCreate(&(streams[n])));
    }

    for (int num = 1; num <= MAX_NUM_STREAMS; ++num)
    {
        timing(d_x, d_y, d_z, num);
    }

    for (int n = 0 ; n < MAX_NUM_STREAMS; ++n)
    {
        CHECK(cudaStreamDestroy(streams[n]));
    }

    free(h_x);
    free(h_y);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const real *d_x, const real *d_y, real *d_z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N1)
    {
        for (int i = 0; i < 100000; ++i)
        {
            d_z[n] = d_x[n] + d_y[n];
        }
    }
}

void timing(const real *d_x, const real *d_y, real *d_z, const int num)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int n = 0; n < num; ++n)
        {
            int offset = n * N1;
            add<<<grid_size, block_size, 0, streams[n]>>>
            (d_x + offset, d_y + offset, d_z + offset);
        }
 
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%g\n", t_ave);
}


