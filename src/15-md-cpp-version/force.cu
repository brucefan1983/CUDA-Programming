#include "force.cuh"
#include "mic.cuh"

void find_force(int N, int MN, Atom *atom)
{
    int *NN = atom->NN;
    int *NL = atom->NL;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    real *pe = atom->pe;
    real *box = atom->box;
    const real epsilon = 1.032e-2;
    const real sigma = 3.405;
    const real cutoff = 10.0;
    const real cutoff_square = cutoff * cutoff;
    const real sigma_3 = sigma * sigma * sigma;
    const real sigma_6 = sigma_3 * sigma_3;
    const real sigma_12 = sigma_6 * sigma_6;
    const real e24s6 = 24.0 * epsilon * sigma_6; 
    const real e48s12 = 48.0 * epsilon * sigma_12;
    const real e4s6 = 4.0 * epsilon * sigma_6; 
    const real e4s12 = 4.0 * epsilon * sigma_12;
    for (int n = 0; n < N; ++n) 
    { 
        fx[n] = fy[n] = fz[n] = pe[n] = 0.0; 
    }
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < NN[i]; k++)
        {
            int j = NL[i * MN + k];
            if (j < i) { continue; }
            real x_ij = x[j] - x[i];
            real y_ij = y[j] - y[i];
            real z_ij = z[j] - z[i];
            apply_mic(box, &x_ij, &y_ij, &z_ij);
            real r2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij;
            if (r2 > cutoff_square) { continue; }
            real r2inv = 1.0 / r2;
            real r4inv = r2inv * r2inv;
            real r6inv = r2inv * r4inv;
            real r8inv = r4inv * r4inv;
            real r12inv = r4inv * r8inv;
            real r14inv = r6inv * r8inv;
            real f_ij = e24s6 * r8inv - e48s12 * r14inv;
            pe[i] += e4s12 * r12inv - e4s6 * r6inv;
            fx[i] += f_ij * x_ij; fx[j] -= f_ij * x_ij;
            fy[i] += f_ij * y_ij; fy[j] -= f_ij * y_ij;
            fz[i] += f_ij * z_ij; fz[j] -= f_ij * z_ij;
        }
    }
}

