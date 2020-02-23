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
    int *NN;
    int *NL;
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
