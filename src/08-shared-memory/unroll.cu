#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int TILE_DIM = 32;
const int BLOCK_Y = 4;

__global__ void transpose(real *A, real *B, int N);
void print_matrix(int N, real *A);

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    int N2 = N * N;
    int grid_size_x = (N - 1) / TILE_DIM + 1;
    int grid_size_y = (N - 1) / TILE_DIM + 1;
    dim3 block_size(TILE_DIM, BLOCK_Y);
    dim3 grid_size(grid_size_x, grid_size_y);

    int M = sizeof(real) * N2;
    real *h_A = (real *)malloc(M);
    real *h_B = (real *)malloc(M);
    for (int n = 0; n < N2; ++n) { h_A[n] = n; }
    real *A, *B;
    CHECK(cudaMalloc(&A, M))
    CHECK(cudaMalloc(&B, M))
    CHECK(cudaMemcpy(A, h_A, M, cudaMemcpyHostToDevice))

    transpose<<<grid_size, block_size>>>(A, B, N);

    CHECK(cudaMemcpy(h_B, B, M, cudaMemcpyDeviceToHost))
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB = transpose(A) =\n");
        print_matrix(N, h_B);
    }

    free(h_A); free(h_B);
    CHECK(cudaFree(A))
    CHECK(cudaFree(B))
    return 0;
}

__global__ void transpose(real *A, real *B, int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM + 1];

    int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int y = 0; y < TILE_DIM; y += BLOCK_Y)
    {
        if (nx < N && (ny + y) < N)
        {
            S[threadIdx.y + y][threadIdx.x] = A[(ny + y) * N + nx];
        }
    }
    __syncthreads();

    nx = blockIdx.x * TILE_DIM + threadIdx.y;
    ny = blockIdx.y * TILE_DIM + threadIdx.x;
    for (int y = 0; y < TILE_DIM; y += BLOCK_Y)
    {
        if ((nx + y) < N && ny < N)
        {
            B[(nx + y) * N + ny] = S[threadIdx.x][threadIdx.y + y];
        }
    }
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

