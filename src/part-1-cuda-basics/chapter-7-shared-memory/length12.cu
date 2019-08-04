#include <stdlib.h> // malloc() and free()
#include <stdio.h> // printf()
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

void __device__ warp_reduce(volatile double *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}

void __global__ get_length_1(double *g_x, double *g_inner, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    __shared__ double s_x[64];
    s_x[tid] = 0.0; 
    if (tid < N) 
    {
        double tmp = g_x[n];
        s_x[tid] += tmp * tmp;
    } 
    __syncthreads();
    if (tid < 32)
    { 
        warp_reduce(s_x, tid); 
    }
    if (tid == 0)
    {
        g_inner[bid] = s_x[0];
    }
}

void __global__ get_length_2(double *g_inner, double *g_length, int N)
{
    int tid = threadIdx.x;
    __shared__ double s_x[1024];
    s_x[tid] = 0.0;
    int number_of_patches = (N - 1) / 1024 + 1; 
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024;
        if (n < N)
        {
            s_x[tid] += g_inner[n];
        }
    }
    __syncthreads();
    if (tid < 512) { s_x[tid] += s_x[tid + 512]; }
    __syncthreads();
    if (tid < 256) { s_x[tid] += s_x[tid + 256]; } 
    __syncthreads();
    if (tid < 128) { s_x[tid] += s_x[tid + 128]; } 
    __syncthreads();
    if (tid <  64) { s_x[tid] += s_x[tid + 64]; }  
    __syncthreads();
    if (tid < 32)
    { 
        warp_reduce(s_x, tid); 
    }
    if (tid == 0)
    {
        g_length[0] = sqrt(s_x[0]);
    }
}

double get_length(double *x, int N)
{
    int grid_size = (N - 1) / 64 + 1;
    double *g_inner;
    cudaMalloc((void**)&g_inner, sizeof(double) * grid_size);
    double *g_length;
    cudaMalloc((void**)&g_length, sizeof(double));
    double *g_x;
    cudaMalloc((void**)&g_x, sizeof(double) * N);
    cudaMemcpy(g_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
    get_length_1<<<grid_size, 64>>>(g_x, g_inner, N);
    get_length_2<<<1, 1024>>>(g_inner, g_length, grid_size);
    double *cpu_length = (double *) malloc(sizeof(double));
    cudaMemcpy(cpu_length, g_length, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(g_inner);
    cudaFree(g_length);
    cudaFree(g_x);
    double length = cpu_length[0];
    free(cpu_length);
    return length;
}

