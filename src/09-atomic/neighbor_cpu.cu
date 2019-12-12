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
const real cutoff = 1.9; // in units of Angstrom
const real cutoff_square = cutoff * cutoff;

void read_xy(real *x, real *y);
void timing(int *NN, int *NL, const real *x, const real *y);
void print_neighbor(const int *NN, const int *NL);

int main(void)
{
    int *NN = (int*) malloc(N * sizeof(int));
    int *NL = (int*) malloc(N * MN * sizeof(int));
    real *x  = (real*) malloc(N * sizeof(real));
    real *y  = (real*) malloc(N * sizeof(real));

    read_xy(x, y);
    timing(NN, NL, x, y);
    print_neighbor(NN, NL);

    free(NN);
    free(NL);
    free(x);
    free(y);
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

void find_neighbor(int *NN, int *NL, const real *x, const real *y)
{
    for (int n = 0; n < N; n++)
    {
        NN[n] = 0;
    }

    for (int n1 = 0; n1 < N; ++n1)
    {
        real x1 = x[n1];
        real y1 = y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            real x12 = x[n2] - x1;
            real y12 = y[n2] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                NL[n1 * MN + NN[n1]++] = n2;
                NL[n2 * MN + NN[n2]++] = n1;
            }
        }
    }
}

void timing(int *NN, int *NL, const real *x, const real *y)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        find_neighbor(NN, NL, x, y);

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

void print_neighbor(const int *NN, const int *NL)
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
            fprintf(fid, " %d", NL[n * MN + k]);
        }
        fprintf(fid, "\n");
    }
    fclose(fid);
}


