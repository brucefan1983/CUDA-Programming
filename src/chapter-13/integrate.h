#pragma once

void equilibration
(
    int Ne, int N, int *NN, int *NL, int MN, 
    double lx, double ly, double lz,
    double T_0, double time_step, double *m, 
    double *fx, double *fy, double *fz, 
    double *vx, double *vy, double *vz, 
    double *x, double *y, double *z
);

void production
(
    int Np, int Ns, int N, int *NN, int *NL, int MN,
    double lx, double ly, double lz,
    double T_0, double time_step, double *m,
    double *fx, double *fy, double *fz,
    double *vx, double *vy, double *vz, 
    double *x, double *y, double *z
);

