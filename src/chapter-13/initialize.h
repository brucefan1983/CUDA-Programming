#pragma once

void initialize_position 
(
    int nx, int ny, int nz,
    double ax, double ay, double az, 
    double *x, double *y, double *z
);
  
void initialize_velocity
(
    int N, double T_0, double *m, 
    double *vx, double *vy, double *vz
);

