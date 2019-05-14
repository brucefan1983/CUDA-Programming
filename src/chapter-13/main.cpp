#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "common.h"
#include "initialize.h"
#include "neighbor.h"
#include "integrate.h"

int main(void)
{
    //srand(time(NULL));
    int nx = 4;
    int ny = nx;
    int nz = ny;
    int N = 4 * nx * ny * nz;
    int Ne = 20000;
    int Np = 20000;
    int Ns = 100;
    int MN = 200;
    double T_0 = 60.0;
    double ax = 5.385;
    double ay = ax;
    double az = ax;
    double lx = ax * nx;
    double ly = ay * ny;
    double lz = az * nz;
    double cutoff = 11.0;
    double time_step = 5.0 / TIME_UNIT_CONVERSION;
    int *NN = (int*) malloc(N * sizeof(int));
    int *NL = (int*) malloc(N * MN * sizeof(int));
    double *m  = (double*) malloc(N * sizeof(double));
    double *x  = (double*) malloc(N * sizeof(double));
    double *y  = (double*) malloc(N * sizeof(double));
    double *z  = (double*) malloc(N * sizeof(double));
    double *vx = (double*) malloc(N * sizeof(double));
    double *vy = (double*) malloc(N * sizeof(double));
    double *vz = (double*) malloc(N * sizeof(double));
    double *fx = (double*) malloc(N * sizeof(double));
    double *fy = (double*) malloc(N * sizeof(double));
    double *fz = (double*) malloc(N * sizeof(double));
    for (int n = 0; n < N; ++n) { m[n] = 40.0; }
    initialize_position(nx, ny, nz, ax, ay, az, x, y, z);
    initialize_velocity(N, T_0, m, vx, vy, vz);
    find_neighbor(N, NN, NL, x, y, z, lx, ly, lz, MN, cutoff);
    equilibration
    (
        Ne, N, NN, NL, MN, lx, ly, lz, T_0, time_step, 
        m, fx, fy, fz, vx, vy, vz, x, y, z
    );
    production
    (
        Np, Ns, N, NN, NL, MN, lx, ly, lz, T_0, time_step, 
        m, fx, fy, fz, vx, vy, vz, x, y, z
    );
    free(NN); free(NL); free(m);  free(x);  free(y);  free(z);
    free(vx); free(vy); free(vz); free(fx); free(fy); free(fz);
    return 0;
}

