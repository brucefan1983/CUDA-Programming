#include "error.cuh"
#include "memory.h"
#include <stdlib.h>

void allocate_memory(int N, int MN, Atom *atom)
{
    CHECK(cudaMallocManaged((void**)&atom->NN, sizeof(int) * N))
    CHECK(cudaMallocManaged((void**)&atom->NL, sizeof(int) * N * MN))
    CHECK(cudaMallocManaged((void**)&atom->m, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->x, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->y, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->z, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->vx, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->vy, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->vz, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->fx, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->fy, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->fz, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->pe, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->ke, sizeof(real) * N))
    CHECK(cudaMallocManaged((void**)&atom->box, sizeof(real) * 6))
}

void deallocate_memory(Atom *atom)
{
    CHECK(cudaFree(atom->NN)) 
    CHECK(cudaFree(atom->NL))
    CHECK(cudaFree(atom->m))
    CHECK(cudaFree(atom->x))
    CHECK(cudaFree(atom->y))
    CHECK(cudaFree(atom->z))
    CHECK(cudaFree(atom->vx))
    CHECK(cudaFree(atom->vy))
    CHECK(cudaFree(atom->vz))
    CHECK(cudaFree(atom->fx))
    CHECK(cudaFree(atom->fy))
    CHECK(cudaFree(atom->fz))
    CHECK(cudaFree(atom->pe))
    CHECK(cudaFree(atom->ke))
    CHECK(cudaFree(atom->box))
}

