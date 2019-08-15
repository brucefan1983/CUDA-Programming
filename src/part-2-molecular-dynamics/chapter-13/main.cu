#include "common.h"
#include "memory.h"
#include "initialize.h"
#include "neighbor.h"
#include "integrate.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv)
{
    //srand(time(NULL));

    int nx = 4;
    if (argc != 2) 
    { 
        printf("Error: ljmd requires one argument\n");
        exit(1);
    }
    else
    {
        nx = atoi(argv[1]);
    }

    int N = 4 * nx * nx * nx;
    int Ne = 2000;
    int Np = 2000;
    int Ns = 100;
    int MN = 128;
    double T_0 = 60.0;
    double ax = 5.385;
    double time_step = 5.0 / TIME_UNIT_CONVERSION;
    Atom atom;
    allocate_memory(N, MN, &atom);
    for (int n = 0; n < N; ++n) { atom.m[n] = 40.0; }
    initialize_position(nx, ax, &atom);
    initialize_velocity(N, T_0, &atom);
    find_neighbor(N, MN, &atom);
    equilibration(Ne, N, MN, T_0, time_step, &atom);
    production(Np, Ns, N, MN, T_0, time_step, &atom);
    deallocate_memory(&atom);
    return 0;
}

