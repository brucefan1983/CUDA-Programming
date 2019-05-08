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
    clock_t time_begin = clock();
    find_neighbor(N, MN, NN, NL, x, y, cutoff);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
    printf("Time used for host function = %f s.\n", time_used);
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

void find_neighbor
(int N, int MN, int *NN, int *NL, double *x, double *y, double cutoff)
{
    double cutoff_square = cutoff * cutoff;
    for (int n = 0; n < N; n++) { NN[n] = 0; }
    for (int n1 = 0; n1 < N - 1; n1++)
    {
        for (int n2 = n1 + 1; n2 < N; n2++)
        {
            double x12 = x[n2] - x[n1];
            double y12 = y[n2] - y[n1];
            double  distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                NL[n1 * MN + NN[n1]++] = n2;
                NL[n2 * MN + NN[n2]++] = n1;
            }
            if (NN[n1] > MN || NN[n2] > MN)
            { printf("Error: MN is too small.\n"); exit(1); }
        }
    }
}

void print_neighbor(int N, int MN, int *NN, int *NL)
{
    FILE *fid = fopen("neighbor3.txt", "w");
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

