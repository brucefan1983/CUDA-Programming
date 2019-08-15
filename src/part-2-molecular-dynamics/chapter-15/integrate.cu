#include "integrate.h"
#include "error.cuh"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static void __global__ gpu_sum_1(int N, double *g_x, double *g_tmp)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    extern __shared__ double s_tmp[];
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
(int N, int number_of_patches, double *g_tmp, double *g_sum)
{
    int tid = threadIdx.x;
    extern __shared__ double s_sum[];
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

static double sum(int N, double *g_x)
{
    const int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    int number_of_patches = (grid_size - 1) / 1024 + 1;

    double *g_tmp;
    CHECK(cudaMalloc((void**)&g_tmp, 
        sizeof(double) * grid_size))
    double *g_sum;
    CHECK(cudaMalloc((void**)&g_sum, sizeof(double)))

    gpu_sum_1
    <<<grid_size, block_size, sizeof(double) * block_size>>>
    (N, g_x, g_tmp);

    gpu_sum_2<<<1, 1024, sizeof(double) * 1024>>>
    (grid_size, number_of_patches, g_tmp, g_sum);

    double *h_sum = (double*) malloc(sizeof(double));
    CHECK(cudaMemcpy(h_sum, g_sum, sizeof(double), 
        cudaMemcpyDeviceToHost))
    double result = h_sum[0];

    CHECK(cudaFree(g_tmp))
    CHECK(cudaFree(g_sum))
    free(h_sum);

    return result;
}

static void __global__ gpu_scale_velocity
(
    int N, double scale_factor, 
    double *g_vx, double *g_vy, double *g_vz
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

static void scale_velocity(int N, double T_0, Atom *atom)
{
    double temperature = sum(N, atom->g_ke) / (1.5 * K_B * N);
    double scale_factor = sqrt(T_0 / temperature);

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_scale_velocity<<<grid_size, block_size>>>
    (N, scale_factor, atom->g_vx, atom->g_vy, atom->g_vz);
}

static void __global__ gpu_integrate
(
    int N, double time_step, double time_step_half,
    double *g_m, double *g_x, double *g_y, double *g_z,
    double *g_vx, double *g_vy, double *g_vz,
    double *g_fx, double *g_fy, double *g_fz, 
    double *g_ke, int flag
)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        double mass = g_m[n];
        double mass_inv = 1.0 / mass;
        double ax = g_fx[n] * mass_inv;
        double ay = g_fy[n] * mass_inv;
        double az = g_fz[n] * mass_inv;
        double vx = g_vx[n];
        double vy = g_vy[n];
        double vz = g_vz[n];

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
(int N, double time_step, Atom *atom, int flag)
{
    double time_step_half = time_step * 0.5;

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
    int Ne, int N, int MN, double T_0, 
    double time_step, Atom *atom
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
    double time_used = (time_finish - time_begin) 
                     / (double) CLOCKS_PER_SEC;
    printf("time used for equilibration = %g s\n", time_used);
}

void production
(
    int Np, int Ns, int N, int MN, double T_0, 
    double time_step, Atom *atom
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
                fid_e, "%20.10e%20.10e\n",
                sum(N, atom->g_ke), sum(N, atom->g_pe)
            );

            CHECK(cudaMemcpy(atom->vx, atom->g_vx, 
                sizeof(double) * N, cudaMemcpyDeviceToHost))
            CHECK(cudaMemcpy(atom->vy, atom->g_vy, 
                sizeof(double) * N, cudaMemcpyDeviceToHost))
            CHECK(cudaMemcpy(atom->vz, atom->g_vz, 
                sizeof(double) * N, cudaMemcpyDeviceToHost))

            for (int n = 0; n < N; ++n)
            {
                double factor = 1.0e5 / TIME_UNIT_CONVERSION;
                fprintf
                (
                    fid_v, "%20.10e%20.10e%20.10e\n", 
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
    double time_used = (time_finish - time_begin) 
                     / (double) CLOCKS_PER_SEC;
    printf("time used for production = %g s\n", time_used);
}
