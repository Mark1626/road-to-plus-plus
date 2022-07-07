#include "mpi.h"
#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
  char buf[256];
  int my_rank, num_procs;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if(my_rank == 0) {
    printf("Number of processes %d\n", num_procs);
  } else {
    printf("Hello from rank %d\n", my_rank);
  }

  MPI_Finalize();
  return 0;
}
