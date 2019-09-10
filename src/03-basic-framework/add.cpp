#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define EPSILON 1.0e-14
void add(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0; y[n] = 2.0; z[n] = 0.0;
    }

    add(x, y, z, N);
    check(z, N);

    free(x); free(y); free(z);
    return 0;
}

void add(double *x, double *y, double *z, int N)
{
    for (int n = 0; n < N; ++n) { z[n] = x[n] + y[n]; }
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 3.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

