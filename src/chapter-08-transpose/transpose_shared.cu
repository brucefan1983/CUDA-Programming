#include "error.cuh"
#include <stdio.h>
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

__global__ void transpose(real *A, real *B, int Nx, int Ny);
void print_matrix(int Ny, int Nx, real *A);

int main(int argc, char **argv)
{
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int block_size_x = atoi(argv[3]);
    int block_size_y = atoi(argv[4]);
    int N = Nx * Ny;
    int grid_size_x = (Nx - 1) / block_size_x + 1;
    int grid_size_y = (Ny - 1) / block_size_y + 1;
    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size(grid_size_x, grid_size_y);

    real *A, *B;
    CHECK(cudaMalloc(&A, sizeof(real) * N))
    CHECK(cudaMalloc(&B, sizeof(real) * N))
    for (int n = 0; n < N; ++n) { A[n] = n; }

    transpose<<<grid_size, block_size>>>(A, B, Nx, Ny);

    CHECK(cudaDeviceSynchronize())
    if (N <= 100)
    {
        printf("A =\n");
        print_matrix(Ny, Nx, A);
        printf("\nB = transpose(A) =\n");
        print_matrix(Nx, Ny, B);
    }

    CHECK(cudaFree(A))
    CHECK(cudaFree(B))
    return 0;
}

__global__ void transpose(real *A, real *B, int Nx, int Ny)
{
    __shared__ real S[32][32];
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < Nx && ny1 < Ny)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * Nx + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < Nx && ny2 < Ny)
    {
        B[nx2 * Ny + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

void print_matrix(int Ny, int Nx, real *A)
{
    for (int ny = 0; ny < Ny; ny++)
    {
        for (int nx = 0; nx < Nx; nx++)
        {
            printf("%g\t", A[ny * Nx + nx]);
        }
        printf("\n");
    }
}

