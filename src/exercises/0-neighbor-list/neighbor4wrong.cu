#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

void read_xy(int N, double *x, double *y);
void find_neighbor
(int N, int MN, int *NN, int *NL, double *x, double *y, double cutoff);
void print_neighbor(int N, int MN, int *NN, int *NL);

int main(void)
{
    int N = 22464; // number of atoms
    int MN = 4; // maximum number of neighbors for each atom
    double cutoff = 1.9; // in units of Angstrom
    int *NN = (int*) malloc(N * sizeof(int));
    int *NL = (int*) malloc(N * MN * sizeof(int));
    double *x  = (double*) malloc(N * sizeof(double));
    double *y  = (double*) malloc(N * sizeof(double));
    read_xy(N, x, y);
    find_neighbor(N, MN, NN, NL, x, y, cutoff);
    print_neighbor(N, MN, NN, NL);
    free(NN); free(NL); free(x); free(y);
    return 0;
}

void read_xy(int N, double *x, double *y)
{
    FILE *fid = fopen("xy.txt", "r");
    if (NULL == fid) { printf("Cannot open xy.in"); exit(1); }
    int count = fscanf(fid, "%d", &N);
    if (count != 1) { printf("Error for reading xy.in"); exit(1); }
    double Lx, Ly;
    count = fscanf(fid, "%lf%lf", &Lx, &Ly);
    if (count != 2) { printf("Error for reading xy.in"); exit(1); }
    for (int n = 0; n < N; ++n)
    {
        count = fscanf(fid, "%lf%lf", &x[n], &y[n]);
        if (count != 2) { printf("Error for reading xy.in"); exit(1); }
    }
    fclose(fid);
}

void __global__ gpu_find_neighbor
(int N, int MN, int *g_NN, int *g_NL, double *g_x, double *g_y, double cutoff2)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x; // from thread to atom
    if (n1 < N)
    {
        g_NN[n1] = 0;
         for (int n2 = n1 + 1; n2 < N; n2++)
        {
            double x12 = g_x[n2] - g_x[n1];
            double y12 = g_y[n2] - g_y[n1];
            double distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff2)
            {
                g_NL[n1 * MN + g_NN[n1]++] = n2;
                g_NL[n2 * MN + g_NN[n2]++] = n1;
            }
        }
    }
}

void find_neighbor
(int N, int MN, int *NN, int *NL, double *x, double *y, double cutoff)
{
    int *g_NN; cudaMalloc((void**)&g_NN, N * sizeof(int));
    int *g_NL; cudaMalloc((void**)&g_NL, N * MN * sizeof(int));
    double *g_x; cudaMalloc((void**)&g_x, N * sizeof(double));
    double *g_y; cudaMalloc((void**)&g_y, N * sizeof(double));
    cudaMemcpy(g_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_find_neighbor<<<grid_size, block_size>>>
    (N, MN, g_NN, g_NL, g_x, g_y, cutoff * cutoff);
    cudaMemcpy(NN, g_NN, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(NL, g_NL, N * MN * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(g_NN); cudaFree(g_NL); cudaFree(g_x); cudaFree(g_y);
}

void print_neighbor(int N, int MN, int *NN, int *NL)
{
    FILE *fid = fopen("neighbor4.txt", "w");
    for (int n = 0; n < N; ++n)
    {
        fprintf(fid, "%d", NN[n]);
        for (int k = 0; k < NN[n]; ++k)
        {
            fprintf(fid, " %d", NL[n * MN + k]);
        }
        fprintf(fid, "\n");
    }
    fclose(fid);
}

