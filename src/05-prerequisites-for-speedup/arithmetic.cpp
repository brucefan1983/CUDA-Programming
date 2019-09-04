#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define EPSILON 1.0e-14
void arithmetic(double *x, double *y, int N);
void check(double *y, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 100.0;
        y[n] = 0.0;
    }
    clock_t time_begin = clock();
    arithmetic(x, y, N);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin)
                     / double(CLOCKS_PER_SEC);
    printf("Time used for host function = %g s.\n", time_used);

    check(y, N);

    free(x); free(y);
    return 0;
}

void arithmetic(double *x, double *y, int N)
{
    for (int n = 0; n < N; ++n)
    {
        double x1 = x[n];
        double x30 = pow(x1, 30.0);
        double sin_x = sin(x30);
        double cos_x = cos(x30);
        y[n] = sin_x * sin_x + cos_x * cos_x;
    }
}

void check(double *y, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(y[n] - 1.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

