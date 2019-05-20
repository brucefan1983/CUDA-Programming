#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define K_B                   8.617343e-5 
#define TIME_UNIT_CONVERSION  1.018051e+1

void apply_mic
(
    double *box, double *x12, double *y12, double *z12
)
{
    if      (*x12 < - box[3]) { *x12 += box[0]; } 
    else if (*x12 > + box[3]) { *x12 -= box[0]; }
    if      (*y12 < - box[4]) { *y12 += box[1]; } 
    else if (*y12 > + box[4]) { *y12 -= box[1]; }
    if      (*z12 < - box[5]) { *z12 += box[2]; } 
    else if (*z12 > + box[5]) { *z12 -= box[2]; }
}

void find_neighbor
(
    int N, int *NN, int *NL, double *x, double *y, double *z, 
    double *box, int MN
)              
{
    double cutoff = 11.0;
    double cutoff_square = cutoff * cutoff;
    for (int n = 0; n < N; n++) {NN[n] = 0;}
    for (int n1 = 0; n1 < N - 1; n1++)
    {  
        for (int n2 = n1 + 1; n2 < N; n2++)
        {   
            double x12 = x[n2] - x[n1];
            double y12 = y[n2] - y[n1];
            double z12 = z[n2] - z[n1];
            apply_mic(box, &x12, &y12, &z12);
            double  d_square = x12*x12 + y12*y12 + z12*z12;
            if (d_square < cutoff_square)
            {        
                NL[n1 * MN + NN[n1]++] = n2;
                NL[n2 * MN + NN[n2]++] = n1;
            }
            if (NN[n1] > MN || NN[n2] > MN)
            {
                printf("Error: MN is too small.\n");
                exit(1);
            }
        }
    } 
}

void initialize_position 
(
    int nx, double ax, double *x, double *y, double *z
)
{
    double x0[4] = {0.0, 0.0, 0.5, 0.5};
    double y0[4] = {0.0, 0.5, 0.0, 0.5}; 
    double z0[4] = {0.0, 0.5, 0.5, 0.0};
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
} 

void scale_velocity
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
     
void initialize_velocity
(
    int N, double T_0, double *m, 
    double *vx, double *vy, double *vz
)
{
    double momentum_average[3] = {0.0, 0.0, 0.0};
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
    scale_velocity(N, T_0, m, vx, vy, vz);
}

void find_force
(
    int N, int *NN, int *NL, int MN, double *box,
    double *x, double *y, double *z, 
    double *fx, double *fy, double *fz,
    double *vx, double *vy, double *vz,
    double *potential
)
{
    const double epsilon = 1.032e-2;
    const double sigma = 3.405;
    const double cutoff = 10.0;
    const double cutoff_square = cutoff * cutoff;
    const double sigma_3 = sigma * sigma * sigma;
    const double sigma_6 = sigma_3 * sigma_3;
    const double sigma_12 = sigma_6 * sigma_6;
    const double e24s6 = 24.0 * epsilon * sigma_6; 
    const double e48s12 = 48.0 * epsilon * sigma_12;
    const double e4s6 = 4.0 * epsilon * sigma_6; 
    const double e4s12 = 4.0 * epsilon * sigma_12;
    *potential = 0.0;
    for (int n = 0; n < N; ++n) { fx[n]=fy[n]=fz[n]=0.0; }
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < NN[i]; k++)
        {
            int j = NL[i * MN + k];
            if (j < i) { continue; }
            double x_ij = x[j] - x[i];
            double y_ij = y[j] - y[i];
            double z_ij = z[j] - z[i];
            apply_mic(box, &x_ij, &y_ij, &z_ij);
            double r_2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij;
            if (r_2 > cutoff_square) { continue; }
            double r_4 = r_2 * r_2;
            double r_6 = r_2 * r_4;
            double r_8 = r_4 * r_4;
            double r_12 = r_4 * r_8;
            double r_14 = r_6 * r_8;
            double f_ij = e24s6 / r_8 - e48s12 / r_14;
            *potential += e4s12 / r_12 - e4s6 / r_6;
            fx[i] += f_ij * x_ij; fx[j] -= f_ij * x_ij;
            fy[i] += f_ij * y_ij; fy[j] -= f_ij * y_ij;
            fz[i] += f_ij * z_ij; fz[j] -= f_ij * z_ij;
        }
    }
}

void integrate
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
    int Ne, int N, int *NN, int *NL, int MN, double *box,
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
            N, NN, NL, MN, box, x, y, z, 
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
    double *box,
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
            N, NN, NL, MN, box, x, y, z, 
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

int main(void)
{
    //srand(time(NULL));
    int nx = 4;
    int N = 4 * nx * nx * nx;
    int Ne = 20000;
    int Np = 20000;
    int Ns = 100;
    int MN = 200;
    double T_0 = 60.0;
    double ax = 5.385;
    double box[6];
    box[0] = ax * nx;
    box[1] = ax * nx;
    box[2] = ax * nx;
    box[3] = box[0] * 0.5;
    box[4] = box[1] * 0.5;
    box[5] = box[2] * 0.5;
    double time_step = 5.0 / TIME_UNIT_CONVERSION;
    int *NN = (int*) malloc(N * sizeof(int));
    int *NL = (int*) malloc(N * MN * sizeof(int));
    double *m  = (double*) malloc(N * sizeof(double));
    double *x  = (double*) malloc(N * sizeof(double));
    double *y  = (double*) malloc(N * sizeof(double));
    double *z  = (double*) malloc(N * sizeof(double));
    double *vx = (double*) malloc(N * sizeof(double));
    double *vy = (double*) malloc(N * sizeof(double));
    double *vz = (double*) malloc(N * sizeof(double));
    double *fx = (double*) malloc(N * sizeof(double));
    double *fy = (double*) malloc(N * sizeof(double));
    double *fz = (double*) malloc(N * sizeof(double));
    for (int n = 0; n < N; ++n) { m[n] = 40.0; }
    initialize_position(nx, ax, x, y, z);
    initialize_velocity(N, T_0, m, vx, vy, vz);
    find_neighbor(N, NN, NL, x, y, z, box, MN);
    equilibration
    (
        Ne, N, NN, NL, MN, box, T_0, time_step, 
        m, fx, fy, fz, vx, vy, vz, x, y, z
    );
    production
    (
        Np, Ns, N, NN, NL, MN, box, T_0, time_step, 
        m, fx, fy, fz, vx, vy, vz, x, y, z
    );
    free(NN); free(NL); free(m);  free(x);  free(y);  free(z);
    free(vx); free(vy); free(vz); free(fx); free(fy); free(fz);
    return 0;
}

