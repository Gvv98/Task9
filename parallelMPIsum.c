#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s N a xval yval\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    double a = atof(argv[2]);
    double xval = atof(argv[3]);
    double yval = atof(argv[4]);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;
    int remainder = N % size;
    int local_N = (rank < remainder) ? chunk + 1 : chunk;
    int offset = (rank < remainder) ? rank * (chunk + 1) : rank * chunk + remainder;

    double *x_local = malloc(local_N * sizeof(double));
    double *y_local = malloc(local_N * sizeof(double));
    double *d_local = malloc(local_N * sizeof(double));

    for (int i = 0; i < local_N; i++) {
        x_local[i] = xval;
        y_local[i] = yval;
    }

    double start_parallel = MPI_Wtime();
    for (int i = 0; i < local_N; i++) {
        d_local[i] = a * x_local[i] + y_local[i];
    }
    double end_parallel = MPI_Wtime();

    // Compute partial sum
    double local_sum = 0.0;
    for (int i = 0; i < local_N; i++) {
        local_sum += d_local[i];
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double *d_global = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        d_global = malloc(N * sizeof(double));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            recvcounts[i] = (i < remainder) ? chunk + 1 : chunk;
        }
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    // Raccolta vettori locali in d_global nel rank 0
    MPI_Gatherv(d_local, local_N, MPI_DOUBLE,
                d_global, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);


    
    // Serial computation on rank 0
    double serial_sum = 0.0;
    double t_serial_start, t_serial_end;
    if (rank == 0) {
        double *x = malloc(N * sizeof(double));
        double *y = malloc(N * sizeof(double));
        double *d_serial = malloc(N * sizeof(double));

        for (int i = 0; i < N; i++) {
            x[i] = xval;
            y[i] = yval;
        }

        t_serial_start = MPI_Wtime();
        for (int i = 0; i < N; i++) {
            d_serial[i] = a * x[i] + y[i];
        }
        t_serial_end = MPI_Wtime();
        
            for (int i = 0; i < N; i++) {
            serial_sum += d_serial[i];
        }

        printf("MPI: first 5 values of d (from rank 0): ");
        for (int i = 0; i < 5 && i < N; i++) {
            printf("%f ", d_global[i]);
        }
        printf("\n");

        printf("Serial sum = %.3f, MPI sum = %.3f\n", serial_sum, global_sum);
        printf("Serial time = %.9f s\n", t_serial_end - t_serial_start);
        printf("MPI time = %.9f s with %d processes\n", end_parallel - start_parallel, size);

        free(x);
        free(y);
        free(d_serial);
    }

    free(x_local);
    free(y_local);
    free(d_local);

    MPI_Finalize();
    return 0;
}
