#include "integrate.h"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static double sum(int N, double *x)
{
    double s = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        s += x[n];
    }
    return s;
}

static void scale_velocity(int N, double T_0, Atom *atom)
{
    double temperature = sum(N, atom->ke) / (1.5 * K_B * N);
    double scale_factor = sqrt(T_0 / temperature);
    for (int n = 0; n < N; ++n)
    { 
        atom->vx[n] *= scale_factor;
        atom->vy[n] *= scale_factor;
        atom->vz[n] *= scale_factor;
    }
}

static void integrate
(int N, double time_step, Atom *atom, int flag)
{
    double *m = atom->m;
    double *x = atom->x;
    double *y = atom->y;
    double *z = atom->z;
    double *vx = atom->vx;
    double *vy = atom->vy;
    double *vz = atom->vz;
    double *fx = atom->fx;
    double *fy = atom->fy;
    double *fz = atom->fz;
    double *ke = atom->ke;
    double time_step_half = time_step * 0.5;
    for (int n = 0; n < N; ++n)
    {
        double mass_inv = 1.0 / m[n];
        double ax = fx[n] * mass_inv;
        double ay = fy[n] * mass_inv;
        double az = fz[n] * mass_inv;
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
            double v2 = vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n];
            ke[n] = m[n] * v2 * 0.5;
        }
    }
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
            fprintf(fid_e, "%20.10e%20.10e\n",
                sum(N, atom->ke), sum(N, atom->pe));
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
