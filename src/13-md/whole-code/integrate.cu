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
    cudaEvent_t start_total, stop_total, start_force, stop_force;
    CHECK(cudaEventCreate(&start_total));
    CHECK(cudaEventCreate(&stop_total));
    CHECK(cudaEventCreate(&start_force));
    CHECK(cudaEventCreate(&stop_force));

    CHECK(cudaEventRecord(start_total));
    cudaEventQuery(start_total);

    FILE *fid_e = fopen("energy.txt", "w");
    for (int step = 0; step < Np; ++step)
    {  
        integrate(N, time_step, atom, 1);

        CHECK(cudaEventRecord(start_force));
        cudaEventQuery(start_force);

        find_force(N, MN, atom);

        CHECK(cudaEventRecord(stop_force));
        CHECK(cudaEventSynchronize(stop_force));
        float t_tmp;
        CHECK(cudaEventElapsedTime(&t_tmp, start_force, stop_force));
        t_force += t_tmp;

        integrate(N, time_step, atom, 2);

        if (0 == step % Ns)
        {
            fprintf(fid_e, "%g %g\n", sum(N, atom->g_ke), sum(N, atom->g_pe));
        }
    }
    fclose(fid_e);

    CHECK(cudaEventRecord(stop_total));
    CHECK(cudaEventSynchronize(stop_total));
    float t_total;
    CHECK(cudaEventElapsedTime(&t_total, start_total, stop_total));
    printf("Time used for production = %g s\n", t_total * 0.001);
    printf("Time used for force part = %g s\n", t_force * 0.001);

    CHECK(cudaEventDestroy(start_total));
    CHECK(cudaEventDestroy(stop_total));
    CHECK(cudaEventDestroy(start_force));
    CHECK(cudaEventDestroy(stop_force));
}


