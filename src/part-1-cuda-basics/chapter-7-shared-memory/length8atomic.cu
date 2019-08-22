#include <stdlib.h>
#include <stdio.h>
#ifdef DOUBLE_PRECISION
    typedef double real;
#else
    typedef float real;
#endif
real get_length(real *x, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(real) * N;
    real *x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
    }
    real length = get_length(x, N);
    printf("length = %g.\n", length);
    free(x);
    return 0;
}

template <int unroll_size>
void __global__ get_length_1
(real *g_x, real *g_inner, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x * unroll_size + tid;
    extern __shared__ real s_inner[];
    s_inner[tid] = 0.0;

    real tmp_sum = 0.0;
    if (n + (unroll_size - 1) * blockDim.x < N)
    {
        real tmp = g_x[n];
        tmp_sum = tmp * tmp;
        if (unroll_size > 1) 
        { 
            tmp = g_x[n + blockDim.x]; 
            tmp_sum += tmp * tmp;
        }
        if (unroll_size > 2) 
        { 
            tmp = g_x[n + 2 * blockDim.x]; 
            tmp_sum += tmp * tmp;
        }
        if (unroll_size > 3) 
        { 
            tmp = g_x[n + 3 * blockDim.x]; 
            tmp_sum += tmp * tmp;
        }
    }
    s_inner[tid] = tmp_sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_inner[tid] += s_inner[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(g_inner, s_inner[0]);
    }
}

real get_length(real *x, int N)
{
    const int block_size = 128;
    const int unroll_size = 4;
    int grid_size = (N - 1) / block_size + 1;
    grid_size = (grid_size - 1) / unroll_size + 1;

    real *cpu_length = (real *) malloc(sizeof(real));
    real *g_length;
    cudaMalloc((void**)&g_length, sizeof(real));
    cpu_length[0] = 0.0;
    cudaMemcpy(g_length, cpu_length, sizeof(real), 
        cudaMemcpyHostToDevice);

    real *g_x;
    cudaMalloc((void**)&g_x, sizeof(real) * N);
    cudaMemcpy(g_x, x, sizeof(real) * N, 
        cudaMemcpyHostToDevice);

    get_length_1<unroll_size>
    <<<grid_size, block_size, sizeof(real) * block_size>>>
    (g_x, g_length, N);

    cudaMemcpy(cpu_length, g_length, sizeof(real), 
        cudaMemcpyDeviceToHost);

    cudaFree(g_length);
    cudaFree(g_x);
    real length = sqrt(cpu_length[0]);
    free(cpu_length);
    return length;
}

