#include "integrate.cuh"
#include "force.cuh"
#include "error.cuh"
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

    clock_t t_total_start = clock();

    FILE *fid = fopen("energy.txt", "w");
    for (int step = 0; step < Np; ++step)
    {
        integrate(N, time_step, atom, 1);

        clock_t t_force_start = clock();

        find_force(N, MN, atom);

        clock_t t_force_stop = clock();

        t_force += float(t_force_stop - t_force_start) / CLOCKS_PER_SEC;

        integrate(N, time_step, atom, 2);

        if (0 == step % Ns)
        {
            fprintf(fid, "%g %g\n", sum(N, atom->ke), sum(N, atom->pe));
        }
    }
    fclose(fid);

    clock_t t_total_stop = clock();

    float t_total = float(t_total_stop - t_total_start) / CLOCKS_PER_SEC;
    printf("Time used for production = %g s\n", t_total);
    printf("Time used for force part = %g s\n", t_force);
}


