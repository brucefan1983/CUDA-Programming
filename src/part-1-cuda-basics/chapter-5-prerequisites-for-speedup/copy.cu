#include "error.cuh"
void __global__ copy(double *x, double *y, int N);

int main(void)
{
    int N = 1 << 27;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
    }
    double *g_x, *g_y;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice))

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    copy<<<grid_size, block_size>>>(g_x, g_y, N);

    free(x);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    return 0;
}

void __global__ copy(double *x, double *y, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    y[n] = x[n];
}

