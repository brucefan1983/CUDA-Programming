#include "error.cuh"
#include <stdio.h>
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

__global__ void transpose(real *A, real *B, int N);
void print_matrix(int N, real *A);

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

    real *A, *B;
    CHECK(cudaMallocManaged(&A, sizeof(real) * N2))
    CHECK(cudaMallocManaged(&B, sizeof(real) * N2))
    for (int n = 0; n < N2; ++n) { A[n] = n; }

    transpose<<<grid_size, block_size>>>(A, B, N);

    CHECK(cudaDeviceSynchronize())
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, A);
        printf("\nB = transpose(A) =\n");
        print_matrix(N, B);
    }

    CHECK(cudaFree(A))
    CHECK(cudaFree(B))
    return 0;
}

__global__ void transpose(real *A, real *B, int N)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N) B[nx * N + ny] = A[ny * N + nx];
}

void print_matrix(int N, real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}

