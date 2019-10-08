#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double reduce(const double *x, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
    }

    clock_t time_begin = clock();
    double sum = reduce(x, N);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) 
        / double(CLOCKS_PER_SEC);
    printf("Time used = %f s.\n", time_used);

    printf("sum = %f.\n", sum);
    free(x);
    return 0;
}

double reduce(const double *x, const int N)
{
    double sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}

