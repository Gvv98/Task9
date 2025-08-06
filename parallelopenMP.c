#include <stdio.h>
#include <stdlib.h>

// OpenMP
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Usage: %s N num_threads a xval yval\n", argv[0]);
        return 1;
    }
    // Here we include our imput
    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    double a = atof(argv[3]);
    double xval = atof(argv[4]);
    double yval = atof(argv[5]);

    //here we allocate memory for x, y, e d/dserial
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double *d_parallel = (double *)malloc(N * sizeof(double));
    double *d_serial = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        x[i] = xval;
        y[i] = yval;
    }

    // Serial computation (old one)
    double t0 = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        d_serial[i] = a * x[i] + y[i];
    }
    double t1 = omp_get_wtime();

    // Parallel computation with OpenMP
    // We split the work in num_threads chunks
    omp_set_num_threads(num_threads);
    double t2 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        d_parallel[i] = a * x[i] + y[i];
    }
    double t3 = omp_get_wtime();

    // Check sums: sum d[i]serial=sum d[i]parallel
    double sum_serial = 0.0, sum_parallel = 0.0;
    for (int i = 0; i < N; i++) {
        sum_serial += d_serial[i];
        sum_parallel += d_parallel[i];
    }

    printf("OpenMP: first ten values of d (using OpenMP): ");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("%f ", d_parallel[i]);
    }
    printf("\n");

    printf("Serial sum = %.3f, OpenMP sum = %.3f\n", sum_serial, sum_parallel);
    printf("Serial time = %.9f s\n", t1 - t0);
    printf("OpenMP time = %.9f s with %d threads\n", t3 - t2, num_threads);

    free(x);
    free(y);
    free(d_parallel);
    free(d_serial);

    return 0;
}
