#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

const double EPSILON = 1.0e-14;
void arithmetic(double *x, const int N);
void check(const double *x, const int N);

int main(void)
{
    const int N = 1000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);

    clock_t time_begin = clock();
    arithmetic(x, N);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
    printf("Time used = %g s.\n", time_used);

    check(x, N);
    free(x);
    return 0;
}

void arithmetic(double *x, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        double a = 0;
        for (int m = 0; m < 1000; ++m)
        {
            a++;
        }
        x[n] = a;
    }
}

void check(const double *y, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(y[n] - 1000.0) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

