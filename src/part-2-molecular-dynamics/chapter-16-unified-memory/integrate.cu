#include "integrate.h"
#include "error.cuh"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static void __global__ gpu_sum_1(int N, real *g_x, real *g_tmp)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    extern __shared__ real s_tmp[];
    s_tmp[tid] = 0.0;

    if (n < N)
    {
        s_tmp[tid] = g_x[n];
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_tmp[tid] += s_tmp[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_tmp[bid] = s_tmp[0];
    }
}

static void __global__ gpu_sum_2
(int N, int number_of_patches, real *g_tmp, real *g_sum)
{
    int tid = threadIdx.x;
    extern __shared__ real s_sum[];
    s_sum[tid] = 0.0;
 
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * blockDim.x;
        if (n < N)
        {
            s_sum[tid] += g_tmp[n];
        }
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_sum[0] = s_sum[0];
    }
}

static real sum(int N, real *g_x)
{
    const int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    int number_of_patches = (grid_size - 1) / 1024 + 1;

    real *g_tmp;
    CHECK(cudaMalloc((void**)&g_tmp, 
        sizeof(real) * grid_size))
    real *g_sum;
    CHECK(cudaMalloc((void**)&g_sum, sizeof(real)))

    gpu_sum_1
    <<<grid_size, block_size, sizeof(real) * block_size>>>
    (N, g_x, g_tmp);

    gpu_sum_2<<<1, 1024, sizeof(real) * 1024>>>
    (grid_size, number_of_patches, g_tmp, g_sum);

    real *h_sum = (real*) malloc(sizeof(real));
    CHECK(cudaMemcpy(h_sum, g_sum, sizeof(real), 
        cudaMemcpyDeviceToHost))
    real result = h_sum[0];

    CHECK(cudaFree(g_tmp))
    CHECK(cudaFree(g_sum))
    free(h_sum);

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
    FILE *fid_v = fopen("velocity.txt", "w");
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

            cudaDeviceSynchronize(); // avoid buss error for K40
            for (int n = 0; n < N; ++n)
            {
                real factor = 1.0e5 / TIME_UNIT_CONVERSION;
                fprintf
                (
                    fid_v, "%g %g %g\n", 
                    atom->vx[n] * factor,
                    atom->vy[n] * factor,
                    atom->vz[n] * factor
                );
            }
        }
    }
    fclose(fid_e);
    fclose(fid_v);
    cudaDeviceSynchronize();
    clock_t time_finish = clock();
    real time_used = (time_finish - time_begin) 
                     / (real) CLOCKS_PER_SEC;
    printf("time used for production = %g s\n", time_used);
}
