#pragma once

#ifdef DOUBLE_PRECISION
    typedef double real;
#else
    typedef float real;
#endif

#define K_B                   8.617343e-5
#define TIME_UNIT_CONVERSION  1.018051e+1

struct Atom
{
    real *m;
    real *x;
    real *y;
    real *z;
    real *vx;
    real *vy;
    real *vz;
    real *fx;
    real *fy;
    real *fz;
    real *pe;
    real *ke;
    real *box;

    int *g_NN;
    int *g_NL;
    real *g_x;
    real *g_y;
    real *g_z;
    real *g_fx;
    real *g_fy;
    real *g_fz;
    real *g_pe;
};

struct Box
{
    real lx;
    real ly;
    real lz;
    real lx2;
    real ly2;
    real lz2;
};
