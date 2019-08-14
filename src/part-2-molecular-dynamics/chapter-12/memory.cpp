#include "memory.h"
#include <stdlib.h>

void allocate_memory(int N, int MN, Atom *atom)
{
    atom->NN = (int*) malloc(N * sizeof(int));
    atom->NL = (int*) malloc(N * MN * sizeof(int));
    atom->m  = (double*) malloc(N * sizeof(double));
    atom->x  = (double*) malloc(N * sizeof(double));
    atom->y  = (double*) malloc(N * sizeof(double));
    atom->z  = (double*) malloc(N * sizeof(double));
    atom->vx = (double*) malloc(N * sizeof(double));
    atom->vy = (double*) malloc(N * sizeof(double));
    atom->vz = (double*) malloc(N * sizeof(double));
    atom->fx = (double*) malloc(N * sizeof(double));
    atom->fy = (double*) malloc(N * sizeof(double));
    atom->fz = (double*) malloc(N * sizeof(double));
    atom->pe = (double*) malloc(N * sizeof(double));
    atom->ke = (double*) malloc(N * sizeof(double));
    atom->box = (double*) malloc(6 * sizeof(double));
}

void deallocate_memory(Atom *atom)
{
    free(atom->NN);
    free(atom->NL);
    free(atom->m);
    free(atom->x);
    free(atom->y);
    free(atom->z);
    free(atom->vx);
    free(atom->vy);
    free(atom->vz);
    free(atom->fx);
    free(atom->fy);
    free(atom->fz);
    free(atom->pe);
    free(atom->ke);
    free(atom->box);
}

