#include "error.cuh"
#include <stdio.h>
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

__global__ void copy(real *A, real *B, int N);

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    int block_size_x = atoi(argv[2]);
    int block_size_y = atoi(argv[3]);
    int N2 = N * N;
    int grid_size_x = (N - 1) / block_size_x + 1;
    int grid_size_y = (N - 1) / block_size_y + 1;
    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size(grid_size_x, grid_size_y);

    int M = sizeof(real) * N2;
    real *h_A = (real *)malloc(M);
    for (int n = 0; n < N2; ++n) { h_A[n] = n; }
    real *A, *B;
    CHECK(cudaMalloc(&A, M))
    CHECK(cudaMalloc(&B, M))
    CHECK(cudaMemcpy(A, h_A, M, cudaMemcpyHostToDevice))

    copy<<<grid_size, block_size>>>(A, B, N);

    free(h_A);
    CHECK(cudaFree(A))
    CHECK(cudaFree(B))
    return 0;
}

__global__ void copy(real *A, real *B, int N)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    int index = ny * N + nx;
    if (nx < N && ny < N) B[index] = A[index];
}

