#include "memory.cuh"
#include <stdlib.h>

void allocate_memory(int N, int MN, Atom *atom)
{
    atom->NN = (int*) malloc(N * sizeof(int));
    atom->NL = (int*) malloc(N * MN * sizeof(int));
    atom->m  = (real*) malloc(N * sizeof(real));
    atom->x  = (real*) malloc(N * sizeof(real));
    atom->y  = (real*) malloc(N * sizeof(real));
    atom->z  = (real*) malloc(N * sizeof(real));
    atom->vx = (real*) malloc(N * sizeof(real));
    atom->vy = (real*) malloc(N * sizeof(real));
    atom->vz = (real*) malloc(N * sizeof(real));
    atom->fx = (real*) malloc(N * sizeof(real));
    atom->fy = (real*) malloc(N * sizeof(real));
    atom->fz = (real*) malloc(N * sizeof(real));
    atom->pe = (real*) malloc(N * sizeof(real));
    atom->ke = (real*) malloc(N * sizeof(real));
    atom->box = (real*) malloc(6 * sizeof(real));
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

