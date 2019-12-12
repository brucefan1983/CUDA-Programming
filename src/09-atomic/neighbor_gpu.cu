#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10; // number of timings
const int N = 22464; // number of atoms
const int MN = 10; // maximum number of neighbors for each atom
const int mem1 = sizeof(int) * N;
const int mem2 = sizeof(int) * N * MN;
const int mem3 = sizeof(real) * N;
const real cutoff = 1.9; // in units of Angstrom
const real cutoff_square = cutoff * cutoff;

void read_xy(real *, real *);
void timing(int *, int *, const real *, const real *, const bool);
void print_neighbor(const int *, const int *, const bool);

int main(void)
{
    int *h_NN = (int*) malloc(mem1);
    int *h_NL = (int*) malloc(mem2);
    real *h_x  = (real*) malloc(mem3);
    real *h_y  = (real*) malloc(mem3);

    read_xy(h_x, h_y);

    int *d_NN, *d_NL;
    real *d_x, *d_y;
    CHECK(cudaMalloc(&d_NN, mem1));
    CHECK(cudaMalloc(&d_NL, mem2));
    CHECK(cudaMalloc(&d_x, mem3));
    CHECK(cudaMalloc(&d_y, mem3));
    CHECK(cudaMemcpy(d_x, h_x, mem3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, mem3, cudaMemcpyHostToDevice));

    printf("\nnot using atomicAdd:\n");
    timing(d_NN, d_NL, d_x, d_y, false);
    printf("\nusing atomicAdd:\n");
    timing(d_NN, d_NL, d_x, d_y, true);

    CHECK(cudaMemcpy(h_NN, d_NN, mem1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_NL, d_NL, mem2, cudaMemcpyDeviceToHost));

    print_neighbor(h_NN, h_NL, true);

    CHECK(cudaFree(d_NN));
    CHECK(cudaFree(d_NL));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(h_NN);
    free(h_NL);
    free(h_x);
    free(h_y);
    return 0;
}

void read_xy(real *x, real *y)
{
    FILE *fid = fopen("xy.txt", "r");
    if (NULL == fid)
    {
        printf("Cannot open xy.in\n");
        exit(1); 
    }

    int N_read;
    int count = fscanf(fid, "%d", &N_read);
    if (count != 1)
    {
        printf("Error for reading xy.in\n");
        exit(1);
    }
    if (N_read != N)
    {
        printf("Error: The N read in is not the N in the code.\n");
        exit(1);
    }  

    double Lx, Ly;
    count = fscanf(fid, "%lf%lf", &Lx, &Ly);
    if (count != 2)
    {
        printf("Error for reading xy.in\n");
        exit(1);
    }

    for (int n = 0; n < N; ++n)
    {
        double x_read, y_read;
        count = fscanf(fid, "%lf%lf", &x_read, &y_read);
        if (count != 2)
        {
            printf("Error for reading xy.in");
            exit(1);
        }
        x[n] = x_read;
        y[n] = y_read;
    }

    fclose(fid);
}

void __global__ find_neighbor_atomic
(int *d_NN, int *d_NL, const real *d_x, const real *d_y)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        d_NN[n1] = 0;
        real x1 = d_x[n1];
        real y1 = d_y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            real x12 = d_x[n2] - x1;
            real y12 = d_y[n2] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                d_NL[n1 * MN + atomicAdd(&d_NN[n1], 1)] = n2;
                d_NL[n2 * MN + atomicAdd(&d_NN[n2], 1)] = n1;
            }
        }
    }
}

void __global__ find_neighbor_no_atomic
(int *d_NN, int *d_NL, const real *d_x, const real *d_y)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int count = 0;
        real x1 = d_x[n1];
        real y1 = d_y[n1];
        for (int n2 = 0; n2 < N; ++n2)
        {
            real x12 = d_x[n2] - x1;
            real y12 = d_y[n2] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if ((n2 != n1) && (distance_square < cutoff_square))
            {
                d_NL[(count++) * N + n1] = n2;
            }
        }
        d_NN[n1] = count;
    }
}  

void timing
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y, 
    const bool atomic
)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        int block_size = 128;
        int grid_size = (N + block_size - 1) / block_size;

        if (atomic)
        {
            find_neighbor_atomic<<<grid_size, block_size>>>
            (d_NN, d_NL, d_x, d_y); 
        }
        else
        {
            find_neighbor_no_atomic<<<grid_size, block_size>>>
            (d_NN, d_NL, d_x, d_y);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

void print_neighbor(const int *NN, const int *NL, const bool atomic)
{
    FILE *fid = fopen("neighbor.txt", "w");
    for (int n = 0; n < N; ++n)
    {
        if (NN[n] > MN)
        {
            printf("Error: MN is too small.\n");
            exit(1);
        }

        fprintf(fid, "%d", NN[n]);
        for (int k = 0; k < NN[n]; ++k)
        {
            int tmp = atomic ? NL[n * MN + k] : NL[k * N + n];
            fprintf(fid, " %d", tmp);
        }
        fprintf(fid, "\n");
    }
    fclose(fid);
}


