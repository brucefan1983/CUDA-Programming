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
    length = get_length(x, N);
    printf("length = %g.\n", length);
    free(x);
    return 0;
}

template <int unroll_size>
void __global__ get_length_1
(double *g_x, double *g_inner, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x * unroll_size + tid;
    extern __shared__ double s_inner[];
    s_inner[tid] = 0.0;

    double tmp_sum = 0.0;
    if (n + (unroll_size - 1) * blockDim.x < N)
    {
        double tmp = g_x[n];
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

    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        if ((tid & ((offset << 1) - 1)) == 0)
        {
            s_inner[tid] += s_inner[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_inner[bid] = s_inner[0];
    }
}

void __global__ get_length_2
(double *g_inner, double *g_length, int N, int number_of_patches)
{
    int tid = threadIdx.x;
    extern __shared__ double s_length[];
    s_length[tid] = 0.0;
 
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * blockDim.x;
        if (n < N)
        {
            s_length[tid] += g_inner[n];
        }
    }
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        if ((tid & ((offset << 1) - 1)) == 0)
        {
            s_length[tid] += s_length[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_length[0] = s_length[0];
    }
}

double get_length(double *x, int N)
{
    const int block_size = 128;
    const int unroll_size = 4;
    int grid_size = (N - 1) / block_size + 1;
    grid_size = (grid_size - 1) / unroll_size + 1;
    int number_of_patches = (grid_size - 1) / 1024 + 1;
    double *g_inner;
    cudaMalloc((void**)&g_inner, sizeof(double) * grid_size);
    double *g_length;
    cudaMalloc((void**)&g_length, sizeof(double));
    double *g_x;
    cudaMalloc((void**)&g_x, sizeof(double) * N);
    cudaMemcpy(g_x, x, sizeof(double) * N, 
        cudaMemcpyHostToDevice);

    get_length_1<unroll_size>
    <<<grid_size, block_size, sizeof(double) * block_size>>>
    (g_x, g_inner, N);

    get_length_2<<<1, 1024, sizeof(double) * 1024>>>
    (g_inner, g_length, grid_size, number_of_patches);

    double *cpu_length = (double *) malloc(sizeof(double));
    cudaMemcpy(cpu_length, g_length, sizeof(double), 
        cudaMemcpyDeviceToHost);
    cudaFree(g_inner);
    cudaFree(g_length);
    cudaFree(g_x);
    double length = sqrt(cpu_length[0]);
    free(cpu_length);
    return length;
}
