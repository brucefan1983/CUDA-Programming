#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;

void timing(real *h_x, real *d_x, const int N, const int method);
real reduce(real *d_x, const int N, const int method);
void __global__ reduce_more(real *d_x, real *d_y, const int N);
void __global__ reduce_atomic(real *d_x, real *d_y, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nusing atomicAdd:\n");
    timing(h_x, d_x, N, 1);
    printf("\nusing two kernels:\n");
    timing(h_x, d_x, N, 0);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void timing(real *h_x, real *d_x, const int N, const int method)
{
    real sum = 0;
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        const int M = sizeof(real) * N;
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        sum = reduce(d_x, N, method); 

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

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
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    printf("sum = %f.\n", sum);
}

real reduce(real *d_x, const int N, const int method)
{
    const int block_size = 1024;
    const int repeat_size = 10;
    int grid_size = (N + block_size - 1) / block_size;
    grid_size = (grid_size + repeat_size - 1) / repeat_size;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * block_size;

    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    if (method == 0)
    {
        reduce_more<<<grid_size, block_size, smem>>>(d_x, d_y, N);
    }
    else
    {
        reduce_atomic<<<grid_size, block_size, smem>>>(d_x, d_y, N);
    }
    
    if (method == 0 && grid_size > 1)
    {
        reduce_more<<<1, block_size, smem>>>(d_y, d_y, grid_size);
    }

    real h_y[1];
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void __global__ reduce_more(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

void __global__ reduce_atomic(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}


