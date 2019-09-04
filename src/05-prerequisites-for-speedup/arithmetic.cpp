#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define EPSILON 1.0e-14
void arithmetic(double *x, int N);
void check(double *x, int N);

int main(void)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);

    clock_t time_begin = clock();
    arithmetic(x, N);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin)
                     / double(CLOCKS_PER_SEC);
    printf("Time used for host function = %g s.\n", time_used);

    check(x, N);
    free(x);
    return 0;
}

void arithmetic(double *x, int N)
{
    for (int n = 0; n < N; ++n)
    {
        double t = pow(2.0, 30.0);
        double sin_t = sin(t);
        double cos_t = cos(t);
        t = sqrt(sin_t * sin_t + cos_t * cos_t);
        t = exp(t);
        t = log(t);
        x[n] = t;
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

