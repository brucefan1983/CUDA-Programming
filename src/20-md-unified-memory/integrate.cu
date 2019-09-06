#include "integrate.h"
#include "error.cuh"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

void __global__ gpu_sum
(int N, int number_of_rounds, real *g_x, real *g_y)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ real s_y[128];

    real y = 0.0;
    int offset = tid + bid * blockDim.x * number_of_rounds;
    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = round * blockDim.x + offset;
        if (n < N) { y += g_x[n]; }
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1)
    {
        if (tid < offset) { s_y[tid] += s_y[tid + offset]; }
        __syncthreads();
    }
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_y[tid] += s_y[tid + offset]; }
        __syncwarp();
    }
    if (tid == 0) { atomicAdd(g_y, s_y[0]); }
}

static real sum(int N, real *g_x)
{
    int block_size = 128;
    int M = (N - 1) / 25600 + 1;
    int grid_size = (N - 1) / (block_size * M) + 1;

    real *sum;
    CHECK(cudaMallocManaged((void**)&sum, sizeof(real)))
#ifndef CONCURRENT
    CHECK(cudaDeviceSynchronize())
#endif
    sum[0] = 0.0;
    
    gpu_sum<<<grid_size, block_size>>>(N, M, g_x, sum);

#ifndef CONCURRENT
    CHECK(cudaDeviceSynchronize())
#endif
    real result = sum[0];
    CHECK(cudaFree(sum))
    return result;
}

static void __global__ gpu_scale_velocity
(
    int N, real scale_factor, 
    real *g_vx, real *g_vy, real *g_vz
)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    { 
        g_vx[n] *= scale_factor;
        g_vy[n] *= scale_factor;
        g_vz[n] *= scale_factor;
    }
}

static void scale_velocity(int N, real T_0, Atom *atom)
{
    real temperature = sum(N, atom->ke) / (1.5 * K_B * N);
    real scale_factor = sqrt(T_0 / temperature);

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_scale_velocity<<<grid_size, block_size>>>
    (N, scale_factor, atom->vx, atom->vy, atom->vz);
}

static void __global__ gpu_integrate
(
    int N, real time_step, real time_step_half,
    real *g_m, real *g_x, real *g_y, real *g_z,
    real *g_vx, real *g_vy, real *g_vz,
    real *g_fx, real *g_fy, real *g_fz, 
    real *g_ke, int flag
)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real mass = g_m[n];
        real mass_inv = 1.0 / mass;
        real ax = g_fx[n] * mass_inv;
        real ay = g_fy[n] * mass_inv;
        real az = g_fz[n] * mass_inv;
        real vx = g_vx[n];
        real vy = g_vy[n];
        real vz = g_vz[n];

        vx += ax * time_step_half;
        vy += ay * time_step_half;
        vz += az * time_step_half;
        g_vx[n] = vx;
        g_vy[n] = vy;
        g_vz[n] = vz;

        if (flag == 1) 
        { 
            g_x[n] += vx * time_step; 
            g_y[n] += vy * time_step; 
            g_z[n] += vz * time_step; 
        }
        else
        {
            g_ke[n] = (vx*vx + vy*vy + vz*vz) * mass * 0.5;
        }
    }
}

static void integrate
(int N, real time_step, Atom *atom, int flag)
{
    real time_step_half = time_step * 0.5;

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_integrate<<<grid_size, block_size>>>
    (
        N, time_step, time_step_half,
        atom->m, atom->x, atom->y, atom->z,
        atom->vx, atom->vy, atom->vz,
        atom->fx, atom->fy, atom->fz, 
        atom->ke, flag
    );
}

void equilibration
(
    int Ne, int N, int MN, real T_0, 
    real time_step, Atom *atom
)
{
    find_force(N, MN, atom);
    cudaDeviceSynchronize();
    clock_t time_begin = clock();
    for (int step = 0; step < Ne; ++step)
    { 
        integrate(N, time_step, atom, 1);
        find_force(N, MN, atom);
        integrate(N, time_step, atom, 2);
        scale_velocity(N, T_0, atom);
    }
    cudaDeviceSynchronize();
    clock_t time_finish = clock();
    real time_used = (time_finish - time_begin) 
                     / (real) CLOCKS_PER_SEC;
    printf("time used for equilibration = %g s\n", time_used);
}

void production
(
    int Np, int Ns, int N, int MN, real T_0, 
    real time_step, Atom *atom
)
{
    cudaDeviceSynchronize();
    clock_t time_begin = clock();
    FILE *fid_e = fopen("energy.txt", "w");
    for (int step = 0; step < Np; ++step)
    {  
        integrate(N, time_step, atom, 1);
        find_force(N, MN, atom);
        integrate(N, time_step, atom, 2);
        if (0 == step % Ns)
        {
            fprintf
            (
                fid_e, "%g %g\n",
                sum(N, atom->ke), sum(N, atom->pe)
            );
        }
    }
    fclose(fid_e);
    cudaDeviceSynchronize();
    clock_t time_finish = clock();
    real time_used = (time_finish - time_begin) 
                     / (real) CLOCKS_PER_SEC;
    printf("time used for production = %g s\n", time_used);
}
