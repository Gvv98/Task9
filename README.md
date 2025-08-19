# Task9

## OpenMP
This program in C performs a parallel computation on $N$-dimensional vectors $$\vec{d}=a\cdot\vec{x}+\vec{y}$$ using OpenMP. The full code is reported in [parallelopenMP.c](https://github.com/Gvv98/Task9/blob/main/parallelopenMP.c) .

The program begins by taking five command-line arguments: the number of elements $$N$$, the number of threads ``num_threads``, a scalar $$a$$, and two values $$xval$$ and $yval$ used to initialize the vectors $$\vec{x}$$ and $$\vec{y}$$. Memory is dynamically allocated for 4 arrays: ``x``, ``y``, ``d_serial``, and ``d_parallel``.  ``x`` and ``y`` are filled with the constant values xval and yval.

The first part of the computation is done serially followewing the standard logic: a for loop calculates each element of the output vector ``d_serial``. The time for this computation is measured using omp_get_wtime():
```
 double t0 = omp_get_wtime();
 for (int i = 0; i < N; i++) {
    d_serial[i] = a * x[i] + y[i];
 }
 double t1 = omp_get_wtime();
```

Then, OpenMP is used to perform the same computation in parallel. The number of threads is set using ``omp_set_num_threads(num_threads)``. The actual parallelization happens through the ``#pragma omp parallel`` for directive, which splits the for loop across the available threads. The output is stored in the vector ``d_parallel``, and the parallel execution time is also measured:
```
omp_set_num_threads(num_threads);
    double t2 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        d_parallel[i] = a * x[i] + y[i];
    }
    double t3 = omp_get_wtime();
```

To validate that the parallelization was correct, the program computes the total sum of elements in both ``d_serial`` and ``d_parallel`` and compares the two. It also prints the first 10 values of ``d_parallel``, along with the times taken for the serial and parallel computations and the number of threads used.





## MPI
This MPI-based program performs a parallel computation on $N$-dimensional vectors as above. The user provides $N$, a scalar $a$, and 2 initial values for $$\vec{x}$$ and $$\vec{y}$$.
The program is executed using the package openmpi, thus install it:
```
yum install -y openmpi openmpi-devel
```
The code is reported in [parallelMPIsum.c](https://github.com/Gvv98/Task9/blob/main/parallelMPIsum.c). Befor startign, check the number of avaible cores using 
```
lscpu
```
An example for a standard computer is:
```
    Thread(s) per core:   2
    Core(s) per socket:   2
```
After choosing the numer of cores to use, compile code with
```bash
mpirun --mca pml ob1 --allow-run-as-root --oversubscribe --use-hwthread-cpus -np <num_processes> ./program N a xval yval
```
where ``<num_processes>`` defines the number of logic cores, ``--allow-run-as-root --oversubscribe --use-hwthread-cpus`` allows to use not only physical cores and ``program`` is the compiled code name. The number of cores for this task define ``size``. Then each of them is assigned a unique identifier (an integer number) called ``rank``, and the total number of processes is retrieved by calling ``MPI-provided`` functions:
```
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```
The vector of length ``N`` is divided among the processes:
```
int chunk = N / size;
int remainder = N % size;
int local_N = (rank < remainder) ? chunk + 1 : chunk;
int offset = (rank < remainder) ? rank * (chunk + 1) : rank * chunk + remainder;

```



The division ensures that each process gets roughly the same number of elements, and if ``N`` is not perfectly divisible, the first few processes receive one extra element. Based on the rank, each process determines how many elements it is responsible for (``local_N``) and where its portion starts (``offset``).

Each process initializes its portion of vectors x and y and performs a local computation for the sum vecotr ``d``:
```
for (int i = 0; i < local_N; i++) {
    x_local[i] = xval;
    y_local[i] = yval;
    d_local[i] = a * x_local[i] + y_local[i];
}
```

The time taken for this computation is measured using MPI's built-in timing functions. After the local computation, each process calculates the sum of its part of the resulting vector:
```
double local_sum = 0.0;
for (int i = 0; i < local_N; i++) {
    local_sum += d_local[i];
}
```

These local sums are combined at process 0 using ``MPI_Reduce``, which performs a global sum. In addition, to reconstruct the full result vector ``d``, the program uses ``MPI_Gatherv``, which gathers all local chunks into ``d_global`` on process 0:
```
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


MPI_Gatherv(d_local, local_N, MPI_DOUBLE,
                d_global, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
```
Once the global vector is reconstructed, process 0 also computes the same operation in serial on the full vectors for comparison. Both the parallel and serial execution times and total sums are printed, along with the first few elements of the result vector. The program concludes by freeing all dynamically allocated memory and calling ``MPI_Finalize``. 


## Comparison

By comparing the computational time across the different approaches, we observe that the parallelized codes tend to be slower when the number of operations $$N$$ is relatively small. As $$N$$ increases, the MPI implementation becomes more efficient, and performance (as expected) improves with the number of MPI processes (and threads in the case of OpenMP). Below, we present some numerical tests showing the execution times for the different approaches as a function of $$N$$ varying the number of threads or MPI processes. For better visual clarity, we include the plots both in linear-log and log-log scale.

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/f21e53e6-3c83-49d4-94b2-48c5c2a3d3f9" />

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/cc2373a4-f447-49c0-bed0-0163f385aeff" />





