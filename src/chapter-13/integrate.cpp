#include "integrate.h"
#include "common.h"
#include "force.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static void scale_velocity
(
    int N, double T_0, double *m, 
    double *vx, double *vy, double *vz
)
{  
    double temperature = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        double v2 = vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n];     
        temperature += m[n] * v2; 
    }
    temperature /= 3.0 * K_B * N;
    double scale_factor = sqrt(T_0 / temperature);
    for (int n = 0; n < N; ++n)
    { 
        vx[n] *= scale_factor;
        vy[n] *= scale_factor;
        vz[n] *= scale_factor;
    }
}

static void integrate
(
    int N, double time_step, double *m, 
    double *fx, double *fy, double *fz, 
    double *vx, double *vy, double *vz, 
    double *x, double *y, double *z, 
    int flag
)
{
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
    int Ne, int N, int *NN, int *NL, int MN, 
    double lx, double ly, double lz,
    double T_0, double time_step, double *m, 
    double *fx, double *fy, double *fz, 
    double *vx, double *vy, double *vz, 
    double *x, double *y, double *z
)
{
    clock_t time_begin = clock();
    double potential;
    for (int step = 0; step < Ne; ++step)
    { 
        integrate
        (
            N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 1
        );
        find_force
        (
            N, NN, NL, MN, lx, ly, lz, x, y, z, 
            fx, fy, fz, vx, vy, vz, &potential
        );
        integrate
        (
            N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 2
        );
        scale_velocity(N, T_0, m, vx, vy, vz);
    } 
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) 
                     / (double) CLOCKS_PER_SEC;
    printf("time use for equilibration = %g s\n", time_used);
}

void production
(
    int Np, int Ns, int N, int *NN, int *NL, int MN,
    double lx, double ly, double lz,
    double T_0, double time_step, double *m,
    double *fx, double *fy, double *fz,
    double *vx, double *vy, double *vz, 
    double *x, double *y, double *z
)
{
    double time_begin = clock();
    FILE *fid_e = fopen("energy.txt", "w");
    FILE *fid_v = fopen("velocity.txt", "w");
    double potential;
    for (int step = 0; step < Np; ++step)
    {  
        integrate
        (
            N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 1
        );
        find_force
        (
            N, NN, NL, MN, lx, ly, lz, x, y, z, 
            fx, fy, fz, vx, vy, vz, &potential
        );
        integrate
        (
            N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 2
        );
        if (0 == step % Ns)
        {
            double ek = 0.0;
            for (int n = 0; n < N; ++n)
            {
                double vx2 = vx[n]; vx2 *= vx2;
                double vy2 = vy[n]; vy2 *= vy2;
                double vz2 = vz[n]; vz2 *= vz2;
                ek += (vx2 + vy2 + vz2) * m[n];
            }
            ek *= 0.5;
            fprintf(fid_e, "%25.15e%25.15e\n", ek, potential);
            for (int n = 0; n < N; ++n)
            {
                double factor = 1.0e5 / TIME_UNIT_CONVERSION;
                fprintf
                (
                    fid_v, "%25.15e%25.15e%25.15e\n", 
                    vx[n] * factor,
                    vy[n] * factor,
                    vz[n] * factor
                );
            }
        }
    }
    fclose(fid_e);
    fclose(fid_v);
    double time_finish = clock();
    double time_used = (time_finish - time_begin) 
                     / (double) CLOCKS_PER_SEC;
    printf("time use for production = %g s\n", time_used);
}

