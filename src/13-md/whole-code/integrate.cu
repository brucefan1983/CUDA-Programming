#include "integrate.h"
#include "error.cuh"
#include "force.h"
#include "reduce.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

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
    real temperature = sum(N, atom->g_ke) / (1.5 * K_B * N);
    real scale_factor = sqrt(T_0 / temperature);

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_scale_velocity<<<grid_size, block_size>>>
    (N, scale_factor, atom->g_vx, atom->g_vy, atom->g_vz);
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

static void integrate(int N, real time_step, Atom *atom, int flag)
{
    real time_step_half = time_step * 0.5;

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_integrate<<<grid_size, block_size>>>
    (
        N, time_step, time_step_half,
        atom->g_m, atom->g_x, atom->g_y, atom->g_z,
        atom->g_vx, atom->g_vy, atom->g_vz,
        atom->g_fx, atom->g_fy, atom->g_fz, 
        atom->g_ke, flag
    );
}

void equilibration
(
    int Ne, int N, int MN, real T_0, 
    real time_step, Atom *atom
)
{
    find_force(N, MN, atom);
    for (int step = 0; step < Ne; ++step)
    { 
        integrate(N, time_step, atom, 1);
        find_force(N, MN, atom);
        integrate(N, time_step, atom, 2);
        scale_velocity(N, T_0, atom);
    } 
}

void production
(
    int Np, int Ns, int N, int MN, real T_0, 
    real time_step, Atom *atom
)
{
    float t_force = 0.0f;
    CHECK(cudaDeviceSynchronize());
    clock_t t_total_start = clock();

    FILE *fid_e = fopen("energy.txt", "w");
    for (int step = 0; step < Np; ++step)
    {  
        integrate(N, time_step, atom, 1);

        CHECK(cudaDeviceSynchronize());
        clock_t t_force_start = clock();

        find_force(N, MN, atom);

        CHECK(cudaDeviceSynchronize());
        clock_t t_force_stop = clock();

        t_force += float(t_force_stop - t_force_start) / CLOCKS_PER_SEC;

        integrate(N, time_step, atom, 2);

        if (0 == step % Ns)
        {
            fprintf(fid_e, "%g %g\n", sum(N, atom->g_ke), sum(N, atom->g_pe));
        }
    }
    fclose(fid_e);

    clock_t t_total_stop = clock();

    float t_total = float(t_total_stop - t_total_start) / CLOCKS_PER_SEC;
    printf("Time used for production = %g s\n", t_total);
    printf("Time used for force part = %g s\n", t_force);
}


