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
    CHECK(cudaMallocManaged(&A, sizeof(real) * N))
    CHECK(cudaMallocManaged(&B, sizeof(real) * N))
    for (int n = 0; n < N; ++n) { A[n] = n; }

    transpose<<<grid_size, block_size>>>(A, B, Nx, Ny);

    CHECK(cudaDeviceSynchronize())
    if (N < 100)
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
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < Nx && ny < Ny) B[nx * Ny + ny] = A[ny * Nx + nx];
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

