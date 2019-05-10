#include <stdlib.h> // malloc() and free()
#include <stdio.h> // printf()
#include <math.h> // sqrt()
double get_length(double *x, int N);

int main(void)
{
    int N = 1000;
    int M = sizeof(double) * N;
    double *x = (double *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
    }
    double length = get_length(x, N);
    printf("length = %g.\n", length);
    free(x);
    return 0;
}

double get_length(double *x, int N)
{
    double length = 0.0;
    for (int n = 0; n < N; ++n)
    {
        length += x[n] * x[n];
    }
    length = sqrt(length);
    return length;
}

