#include "initialize.h"
#include "common.h"
#include <stdlib.h>
#include <math.h>

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

void initialize_position 
(
    int nx, int ny, int nz,
    double ax, double ay, double az, 
    double *x, double *y, double *z
)
{
    double x0[4] = {0.0, 0.0, 0.5, 0.5};
    double y0[4] = {0.0, 0.5, 0.0, 0.5}; 
    double z0[4] = {0.0, 0.5, 0.5, 0.0};
    int n = 0;
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                for (int i = 0; i < 4; ++i)
                {
                    x[n] = (ix + x0[i]) * ax;
                    y[n] = (iy + y0[i]) * ay;
                    z[n] = (iz + z0[i]) * az;
                    n++;
                }
            }
        }
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

