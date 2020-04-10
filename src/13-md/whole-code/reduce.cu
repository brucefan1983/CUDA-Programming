#include "reduce.h"
#include "error.cuh"
#include <cooperative_groups.h>
using namespace cooperative_groups;
const int block_size = 128;

void __global__ reduce_cp(const real *d_x, real *d_y, const int N)
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

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        d_y[bid] = y;
    }
}

__device__ real static_y[block_size];

real sum(const int N, const real *d_x)
{
    real *d_y;
    CHECK(cudaGetSymbolAddress((void**)&d_y, static_y));
    
    const int smem = sizeof(real) * block_size;

    reduce_cp<<<block_size, block_size, smem>>>(d_x, d_y, N);
    reduce_cp<<<1, block_size, smem>>>(d_y, d_y, block_size);

    real h_y[1] = {0};
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));

    return h_y[0];
}


