#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "common.cuh"
#include "memory.cuh"
#include "initialize.cuh"
#include "neighbor.cuh"
#include "integrate.cuh"

int main(void)
{
    //srand(time(NULL));
    int nx = 4;
    int N = 4 * nx * nx * nx;
    int Ne = 20000;
    int Np = 20000;
    int Ns = 100;
    int MN = 200;
    double T_0 = 60.0;
    double ax = 5.385;
    double box[6];
    box[0] = ax * nx;
    box[1] = ax * nx;
    box[2] = ax * nx;
    box[3] = box[0] * 0.5;
    box[4] = box[1] * 0.5;
    box[5] = box[2] * 0.5;
    double time_step = 5.0 / TIME_UNIT_CONVERSION;
    Atom atom;
    allocate_memory(N, MN, &atom);
    for (int n = 0; n < N; ++n) { atom.m[n] = 40.0; }
    initialize_position(nx, ax, &atom);
    initialize_velocity(N, T_0, &atom);
    find_neighbor(N, MN, box, &atom);
    equilibration(Ne, N, MN, box, T_0, time_step, &atom);
    production(Np, Ns, N, MN, box, T_0, time_step, &atom);
    deallocate_memory(&atom);
    return 0;
}

