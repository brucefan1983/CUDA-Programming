#pragma once

#define K_B                   8.617343e-5
#define TIME_UNIT_CONVERSION  1.018051e+1

struct Atom
{
    int *NN;
    int *NL;
    double *m;
    double *x;
    double *y;
    double *z;
    double *vx;
    double *vy;
    double *vz;
    double *fx;
    double *fy;
    double *fz;
    double *pe;
    double *box;

    int *g_NN;
    int *g_NL;
    double *g_x;
    double *g_y;
    double *g_z;
    double *g_fx;
    double *g_fy;
    double *g_fz;
    double *g_pe;
    double *g_box;
};
