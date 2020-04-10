#include "initialize.h"
#include "error.cuh"
#include <stdlib.h>
#include <math.h>

static void scale_velocity(int N, real T_0, Atom *atom)
{
    real *m = atom->m;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real temperature = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        real v2 = vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n];     
        temperature += m[n] * v2; 
    }
    temperature /= 3.0 * K_B * N;
    real scale_factor = sqrt(T_0 / temperature);
    for (int n = 0; n < N; ++n)
    { 
        vx[n] *= scale_factor;
        vy[n] *= scale_factor;
        vz[n] *= scale_factor;
    }
}

void initialize_position(int nx, real ax, Atom *atom)
{
    atom->box[0] = ax * nx;
    atom->box[1] = ax * nx;
    atom->box[2] = ax * nx;
    atom->box[3] = atom->box[0] * 0.5;
    atom->box[4] = atom->box[1] * 0.5;
    atom->box[5] = atom->box[2] * 0.5;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real x0[4] = {0.0, 0.0, 0.5, 0.5};
    real y0[4] = {0.0, 0.5, 0.0, 0.5}; 
    real z0[4] = {0.0, 0.5, 0.5, 0.0};
    int n = 0;
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < nx; ++iy)
        {
            for (int iz = 0; iz < nx; ++iz)
            {
                for (int i = 0; i < 4; ++i)
                {
                    x[n] = (ix + x0[i]) * ax;
                    y[n] = (iy + y0[i]) * ax;
                    z[n] = (iz + z0[i]) * ax;
                    n++;
                }
            }
        }
    }

    int m1 = sizeof(real) * 4 * nx * nx * nx;
    CHECK(cudaMemcpy(atom->g_x, atom->x, m1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_y, atom->y, m1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_z, atom->z, m1, cudaMemcpyHostToDevice));
}

void initialize_velocity(int N, real T_0, Atom *atom)
{
    real *m = atom->m;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real momentum_average[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < N; ++n)
    { 
        vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;    
        
        momentum_average[0] += m[n] * vx[n] / N;
        momentum_average[1] += m[n] * vy[n] / N;
        momentum_average[2] += m[n] * vz[n] / N;
    } 
    for (int n = 0; n < N; ++n) 
    { 
        vx[n] -= momentum_average[0] / m[n];
        vy[n] -= momentum_average[1] / m[n];
        vz[n] -= momentum_average[2] / m[n]; 
    }
    scale_velocity(N, T_0, atom);

    CHECK(cudaMemcpy(atom->g_m, atom->m, sizeof(real) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_vx, atom->vx, sizeof(real) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_vy, atom->vy, sizeof(real) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_vz, atom->vz, sizeof(real) * N, cudaMemcpyHostToDevice));
}

