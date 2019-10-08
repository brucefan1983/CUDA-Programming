#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

const double EPSILON = 1.0e-14;
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
        y[n] = 2.0;
    }

    clock_t time_begin = clock();
    add(x, y, z, N);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
    printf("Time used = %g s.\n", time_used);

    check(z, N);
    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - 3.0) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

