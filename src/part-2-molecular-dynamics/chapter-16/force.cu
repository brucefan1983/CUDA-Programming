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

#ifdef USE_BLOCK_SCHEME
static void __global__ gpu_find_force_block
(
    LJ lj, int N, int MN, int number_of_patches,
    int *g_NN, int *g_NL, Box box,
#ifdef USE_LDG
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z,
#else
    double *g_x, 
    double *g_y, 
    double *g_z,
#endif 
    double *g_fx, double *g_fy, double *g_fz, double *g_pe
)
{
    __shared__ double s_fx[128];
    __shared__ double s_fy[128];
    __shared__ double s_fz[128];
    __shared__ double s_pe[128];

    int tid = threadIdx.x;
    s_fx[tid] = 0.0;
    s_fy[tid] = 0.0;
    s_fz[tid] = 0.0;
    s_pe[tid] = 0.0;

    int i = blockIdx.x;
    int NN = g_NN[i];
#ifdef USE_LDG
    double x_i = __ldg(g_x + i);
    double y_i = __ldg(g_y + i);
    double z_i = __ldg(g_z + i);
#else
    double x_i = g_x[i];
    double y_i = g_y[i];
    double z_i = g_z[i];
#endif

    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int k = tid + blockDim.x * patch;
        if (k >= NN) { continue; }

        int j = g_NL[i * MN + k];
#ifdef USE_LDG
        double x_ij  = __ldg(g_x + j) - x_i;
        double y_ij  = __ldg(g_y + j) - y_i;
        double z_ij  = __ldg(g_z + j) - z_i;
#else
        double x_ij  = g_x[j] - x_i;
        double y_ij  = g_y[j] - y_i;
        double z_ij  = g_z[j] - z_i;
#endif
        apply_mic(box, &x_ij, &y_ij, &z_ij);
        double r2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij;
        if (r2 > lj.cutoff2) { continue; }

        double r2inv = 1.0 / r2;
        double r4inv = r2inv * r2inv;
        double r6inv = r2inv * r4inv;
        double r8inv = r4inv * r4inv;
        double r12inv = r4inv * r8inv;
        double r14inv = r6inv * r8inv;
        double f_ij = lj.e24s6 * r8inv - lj.e48s12 * r14inv;
        s_pe[tid] += lj.e4s12 * r12inv - lj.e4s6 * r6inv;
        s_fx[tid] += f_ij * x_ij;
        s_fy[tid] += f_ij * y_ij;
        s_fz[tid] += f_ij * z_ij;
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_fx[tid] += s_fx[tid + offset];
            s_fy[tid] += s_fy[tid + offset];
            s_fz[tid] += s_fz[tid + offset];
            s_pe[tid] += s_pe[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_fx[i] = s_fx[0];
        g_fy[i] = s_fy[0];
        g_fz[i] = s_fz[0];
        g_pe[i] = s_pe[0] * 0.5;
    }
}

#else

static void __global__ gpu_find_force
(
    LJ lj, int N, int *g_NN, int *g_NL, Box box,
#ifdef USE_LDG
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z,
#else
    double *g_x, 
    double *g_y, 
    double *g_z,
#endif 
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
#ifdef USE_LDG
        double x_i = __ldg(g_x + i);
        double y_i = __ldg(g_y + i);
        double z_i = __ldg(g_z + i);
#else
        double x_i = g_x[i];
        double y_i = g_y[i];
        double z_i = g_z[i];
#endif
        for (int k = 0; k < NN; ++k)
        {   
            int j = g_NL[i + N * k];
#ifdef USE_LDG
            double x_ij  = __ldg(g_x + j) - x_i;
            double y_ij  = __ldg(g_y + j) - y_i;
            double z_ij  = __ldg(g_z + j) - z_i;
#else
            double x_ij  = g_x[j] - x_i;
            double y_ij  = g_y[j] - y_i;
            double z_ij  = g_z[j] - z_i;
#endif
            apply_mic(box, &x_ij, &y_ij, &z_ij);
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
#endif

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

    Box box;
    box.lx = atom->box[0];
    box.ly = atom->box[1];
    box.lz = atom->box[2];
    box.lx2 = atom->box[3];
    box.ly2 = atom->box[4];
    box.lz2 = atom->box[5];

    int block_size = 128;
#ifdef USE_BLOCK_SCHEME 
    int grid_size = N;
    int number_of_patches = (MN - 1) / block_size + 1;
    int smem = sizeof(double) * block_size;
    gpu_find_force_block<<<grid_size, block_size, smem>>>
    (
        lj, N, MN, number_of_patches, 
        atom->g_NN, atom->g_NL, box,
        atom->g_x, atom->g_y, atom->g_z,
        atom->g_fx, atom->g_fy, atom->g_fz, atom->g_pe
    );
#else
    int grid_size = (N - 1) / block_size + 1;
    gpu_find_force<<<grid_size, block_size>>>
    (
        lj, N,  atom->g_NN, atom->g_NL, box,
        atom->g_x, atom->g_y, atom->g_z,
        atom->g_fx, atom->g_fy, atom->g_fz, atom->g_pe
    );
#endif
}

