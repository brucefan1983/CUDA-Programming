#include "force.h"
#include "mic.h"

void find_force
(int N, int MN, Atom *atom, double *potential)
{
    int *NN = atom->NN;
    int *NL = atom->NL;
    double *x = atom->x;
    double *y = atom->y;
    double *z = atom->z;
    double *fx = atom->fx;
    double *fy = atom->fy;
    double *fz = atom->fz;
    double *box = atom->box;
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
            double r2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij;
            if (r2 > cutoff_square) { continue; }
            double r2inv = 1.0 / r2;
            double r4inv = r2inv * r2inv;
            double r6inv = r2inv * r4inv;
            double r8inv = r4inv * r4inv;
            double r12inv = r4inv * r8inv;
            double r14inv = r6inv * r8inv;
            double f_ij = e24s6 * r8inv - e48s12 * r14inv;
            *potential += e4s12 * r12inv - e4s6 * r6inv;
            fx[i] += f_ij * x_ij; fx[j] -= f_ij * x_ij;
            fy[i] += f_ij * y_ij; fy[j] -= f_ij * y_ij;
            fz[i] += f_ij * z_ij; fz[j] -= f_ij * z_ij;
        }
    }
}

