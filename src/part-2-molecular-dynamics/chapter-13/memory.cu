#include "error.cuh"
#include "memory.h"
#include <stdlib.h>

void allocate_memory(int N, int MN, Atom *atom)
{
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
    atom->box = (double*) malloc(6 * sizeof(double));

    CHECK(cudaMalloc((void**)&atom->g_NN, sizeof(int) * N))
    CHECK(cudaMalloc((void**)&atom->g_NL, sizeof(int) * N * MN))
    CHECK(cudaMalloc((void**)&atom->g_x, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_y, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_z, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_fx, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_fy, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_fz, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_pe, sizeof(double) * N))
    CHECK(cudaMalloc((void**)&atom->g_box, sizeof(double) * 6))
}

void deallocate_memory(Atom *atom)
{
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
    free(atom->box);

    CHECK(cudaFree(atom->g_NN)) 
    CHECK(cudaFree(atom->g_NL))
    CHECK(cudaFree(atom->g_x))
    CHECK(cudaFree(atom->g_y))
    CHECK(cudaFree(atom->g_z))
    CHECK(cudaFree(atom->g_fx))
    CHECK(cudaFree(atom->g_fy))
    CHECK(cudaFree(atom->g_fz))
    CHECK(cudaFree(atom->g_pe))
    CHECK(cudaFree(atom->g_box))
}

