#include "neighbor.h"
#include "mic.h"
#include <stdio.h>
#include <stdlib.h>

void find_neighbor
(
    int N, int *NN, int *NL, double *x, double *y, double *z, 
    double lx, double ly, double lz, int MN, double cutoff
)              
{
    double lxh = lx * 0.5;
    double lyh = ly * 0.5;
    double lzh = lz * 0.5; 
    double cutoff_square = cutoff * cutoff;
    for (int n = 0; n < N; n++) {NN[n] = 0;}
    for (int n1 = 0; n1 < N - 1; n1++)
    {  
        for (int n2 = n1 + 1; n2 < N; n2++)
        {   
            double x12 = x[n2] - x[n1];
            double y12 = y[n2] - y[n1];
            double z12 = z[n2] - z[n1];
            apply_mic
            (
                lx, ly, lz, lxh, lyh, lzh, &x12, &y12, &z12
            );
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

