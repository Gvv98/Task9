# Task9


<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/0a916a60-f681-414a-861e-a0ef2bd01de8" />

## OpenMP




## MPI
This MPI-based program performs a parallel computation on $N$-dimensional vectors $\vec{d}=a\cdot\vec{x}+\vec{y}$. The user provides $N$, a scalar $a$, and 2 initial values for $\vec{x}$ and $\vec{y}. First, we will use the packe
The program is executed using the package openmpi, thus install it:
```
yum install -y openmpi openmpi-devel
```
The code is reported below. Copy it and run the compiled code with
```bash
mpirun -np <num_processes> ./program N a xval yval
```
where ``<num_processes>`` defines the number of logic cores. To check the numeber of avaible cores use
```
lscpu
```
An example for a standard computer is:
```
    Thread(s) per core:   2
    Core(s) per socket:   2
```
After choosing the number of cores for this task (``size``), each of them is assigned a unique identifier (an integer number) called ``rank``, and the total number of processes is retrieved by calling ``MPI-provided`` functions:
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
