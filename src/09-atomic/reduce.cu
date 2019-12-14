#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const int MAX_THREAD = 1024;
const int NUM_ROUNDS = 10;

void timing(const real *d_x, const bool atomic);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing two kernels:\n");
    timing(d_x, false);
    printf("\nusing atomicAdd:\n");
    timing(d_x, true);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

template<bool using_atomic>
void __global__ reduce(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    for (int n = bid * blockDim.x + tid; n < N; n += blockDim.x * gridDim.x)
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
        if (using_atomic)
        {
            atomicAdd(d_y, s_y[0]);
        }
        else
        {
            d_y[bid] = s_y[0];
        }
    }
}

real reduce(const real *d_x, const bool atomic)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = (grid_size + NUM_ROUNDS - 1) / NUM_ROUNDS;
    const int ymem = atomic ? sizeof(real) : sizeof(real) * grid_size;
    const int smem1 = sizeof(real) * BLOCK_SIZE;
    const int smem2 = sizeof(real) * MAX_THREAD;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    if (atomic)
    {
        CHECK(cudaMemcpy(d_y, h_y, ymem, cudaMemcpyHostToDevice));
        reduce<true><<<grid_size, BLOCK_SIZE, smem1>>>(d_x, d_y, N);
    }
    else
    {
        reduce<false><<<grid_size, BLOCK_SIZE, smem1>>>(d_x, d_y, N);
    }

    if (!atomic && grid_size > 1)
    {
        reduce<false><<<1, MAX_THREAD, smem2>>>(d_y, d_y, grid_size);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x, const bool atomic)
{
    real sum = 0;
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        sum = reduce(d_x, atomic); 

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


