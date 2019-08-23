#include "neighbor.h"
#include "mic.h"
#include <stdio.h>
#include <stdlib.h>

static void __global__ gpu_find_neighbor
(
    int N, int *g_NN, int *g_NL, double *g_box, 
    double *g_x, double *g_y, double *g_z, double cutoff2
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int count = 0;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        for (int n2 = 0; n2 < N; n2++)
        {
            double x12 = g_x[n2] - x1;
            double y12 = g_y[n2] - y1;
            double z12 = g_z[n2] - z1;
            apply_mic(g_box, &x12, &y12, &z12);
            double d12_square = x12*x12 + y12*y12 + z12*z12;
            if ((n2 != n1) && (d12_square < cutoff2))
            {
                g_NL[count++ * N + n1] = n2;
            }
        }
        g_NN[n1] = count;
    }
}

void find_neighbor(int N, int MN, Atom *atom)
{
    double cutoff = 10.0;
    double cutoff2 = cutoff * cutoff;

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    gpu_find_neighbor<<<grid_size, block_size>>>
    (
        N, atom->g_NN, atom->g_NL, atom->g_box,
        atom->g_x, atom->g_y, atom->g_z, cutoff2
    );
}

