#include "error.cuh"
#include "force.h"
#include "mic.h"

struct LJ
{
    double cutoff2;
    double e24s6; 
    double e48s12;
    double e4s6;
    double e4s12;
};

static void __global__ gpu_find_force
(
    LJ lj, int N, int *g_NN, int *g_NL, double *g_box,
    double *g_x, double *g_y, double *g_z,
    double *g_fx, double *g_fy, double *g_fz, double *g_pe
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        double potential = 0.0;
        int NN = g_NN[i];
        double x_i = g_x[i];
        double y_i = g_y[i];
        double z_i = g_z[i];
        for (int k = 0; k < NN; ++k)
        {   
            int j = g_NL[i + N * k];
            double x_ij  = g_x[j] - x_i;
            double y_ij  = g_y[j] - y_i;
            double z_ij  = g_z[j] - z_i;
            apply_mic(g_box, &x_ij, &y_ij, &z_ij);
            double r2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij;
            if (r2 > lj.cutoff2) { continue; }
            double r2inv = 1.0 / r2;
            double r4inv = r2inv * r2inv;
            double r6inv = r2inv * r4inv;
            double r8inv = r4inv * r4inv;
            double r12inv = r4inv * r8inv;
            double r14inv = r6inv * r8inv;
            double f_ij = lj.e24s6 * r8inv - lj.e48s12 * r14inv;
            potential += lj.e4s12 * r12inv - lj.e4s6 * r6inv;
            fx += f_ij * x_ij;
            fy += f_ij * y_ij;
            fz += f_ij * z_ij;
        }
        g_fx[i] = fx; 
        g_fy[i] = fy; 
        g_fz[i] = fz; 
        g_pe[i] = potential * 0.5;
    }
}

void find_force(int N, int MN, Atom *atom)
{
    const double epsilon = 1.032e-2;
    const double sigma = 3.405;
    const double cutoff = 10.0;
    const double cutoff2 = cutoff * cutoff;
    const double sigma_3 = sigma * sigma * sigma;
    const double sigma_6 = sigma_3 * sigma_3;
    const double sigma_12 = sigma_6 * sigma_6;
    const double e24s6 = 24.0 * epsilon * sigma_6; 
    const double e48s12 = 48.0 * epsilon * sigma_12;
    const double e4s6 = 4.0 * epsilon * sigma_6;
    const double e4s12 = 4.0 * epsilon * sigma_12;
    LJ lj;
    lj.cutoff2 = cutoff2;
    lj.e24s6 = e24s6;
    lj.e48s12 = e48s12;
    lj.e4s6 = e4s6;
    lj.e4s12 = e4s12;

    int m = sizeof(double) * N;
    CHECK(cudaMemcpy(atom->g_x, atom->x, m, 
        cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(atom->g_y, atom->y, m, 
        cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(atom->g_z, atom->z, m, 
        cudaMemcpyHostToDevice))

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_find_force<<<grid_size, block_size>>>
    (
        lj, N, atom->g_NN, atom->g_NL, atom->g_box,
        atom->g_x, atom->g_y, atom->g_z,
        atom->g_fx, atom->g_fy, atom->g_fz, atom->g_pe
    );

    CHECK(cudaMemcpy(atom->fx, atom->g_fx, m, 
        cudaMemcpyDeviceToHost))
    CHECK(cudaMemcpy(atom->fy, atom->g_fy, m, 
        cudaMemcpyDeviceToHost))
    CHECK(cudaMemcpy(atom->fz, atom->g_fz, m, 
        cudaMemcpyDeviceToHost))
    CHECK(cudaMemcpy(atom->pe, atom->g_pe, m, 
        cudaMemcpyDeviceToHost))
}

