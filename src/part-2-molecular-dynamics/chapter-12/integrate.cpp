#include "integrate.h"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static void find_ek(int N, double T_0, Atom *atom, double *ek)
{
    *ek = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        double v2 = atom->vx[n]*atom->vx[n] 
                  + atom->vy[n]*atom->vy[n] 
                  + atom->vz[n]*atom->vz[n];     
        *ek += atom->m[n] * v2; 
    }
    *ek *= 0.5;
}

static void scale_velocity(int N, double T_0, Atom *atom)
{
    double ek = 0.0;
    find_ek(N, T_0, atom, &ek);
    double temperature = ek /= (1.5 * K_B * N);
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
    }
}

void equilibration
(
    int Ne, int N, int MN, double T_0, 
    double time_step, Atom *atom
)
{
    clock_t time_begin = clock();
    double potential;
    for (int step = 0; step < Ne; ++step)
    { 
        integrate(N, time_step, atom, 1);
        find_force(N, MN, atom, &potential);
        integrate(N, time_step, atom, 2);
        scale_velocity(N, T_0, atom);
    } 
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
    double time_begin = clock();
    FILE *fid_e = fopen("energy.txt", "w");
    FILE *fid_v = fopen("velocity.txt", "w");
    double potential;
    for (int step = 0; step < Np; ++step)
    {  
        integrate(N, time_step, atom, 1);
        find_force(N, MN, atom, &potential);
        integrate(N, time_step, atom, 2);
        if (0 == step % Ns)
        {
            double ek = 0.0;
            find_ek(N, T_0, atom, &ek);
            fprintf(fid_e, "%20.10e%20.10e\n", ek, potential);
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
    double time_finish = clock();
    double time_used = (time_finish - time_begin) 
                     / (double) CLOCKS_PER_SEC;
    printf("time used for production = %g s\n", time_used);
}

