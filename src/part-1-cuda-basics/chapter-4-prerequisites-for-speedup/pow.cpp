#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define EPSILON 1.0e-14
void power(double *x, double *y, double *z, int N);
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

    clock_t time_begin = clock();
    power(x, y, z, N);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin)
        / double(CLOCKS_PER_SEC);
    printf("Time used for host function = %f s.\n", time_used);

    check(z, N);
    free(x); free(y); free(z);
    return 0;
}

void power(double *x, double *y, double *z, int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = pow(x[n], y[n]);
    }
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 1.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

