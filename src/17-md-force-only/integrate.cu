#include "integrate.h"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static real sum(int N, real *x)
{
    real s = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        s += x[n];
    }
    return s;
}

static void scale_velocity(int N, real T_0, Atom *atom)
{
    real temperature = sum(N, atom->ke) / (1.5 * K_B * N);
    real scale_factor = sqrt(T_0 / temperature);
    for (int n = 0; n < N; ++n)
    { 
        atom->vx[n] *= scale_factor;
        atom->vy[n] *= scale_factor;
        atom->vz[n] *= scale_factor;
    }
}

static void integrate
(int N, real time_step, Atom *atom, int flag)
{
    real *m = atom->m;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    real *ke = atom->ke;
    real time_step_half = time_step * 0.5;
    for (int n = 0; n < N; ++n)
    {
        real mass_inv = 1.0 / m[n];
        real ax = fx[n] * mass_inv;
        real ay = fy[n] * mass_inv;
        real az = fz[n] * mass_inv;
        vx[n] += ax * time_step_half;
        vy[n] += ay * time_step_half;
        vz[n] += az * time_step_half;
        if (flag == 1) 
        { 
            x[n] += vx[n] * time_step; 
            y[n] += vy[n] * time_step; 
            z[n] += vz[n] * time_step; 
        }
        else
        {
            real v2 = vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n];
            ke[n] = m[n] * v2 * 0.5;
        }
    }
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
