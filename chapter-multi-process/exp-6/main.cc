// Example which uses MPI + OpenMP + OpenMP Offloading to GPU, the dream setup for Data Intensive Computing
// mpic++ -o DIC_Dream DIC_Dream.cc -lboost_mpi -lboost_serialization -fopenmp
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdio>
#include <iostream>
#include <omp.h>

namespace bmpi = boost::mpi;

inline bool isMaster(int rank) {
  return rank == 0;
}

int main(int argc, char **argv) {
  bmpi::environment env(argc, argv, bmpi::threading::funneled);
  bmpi::communicator comm;

  // Spawn GPU tasks from MPI master
  if (isMaster(comm.rank())) {
    int core = 0;
    int thread = 0;
    #pragma omp parallel
    {
      // Master thread of MPI master 
      #pragma omp master
      {
        #pragma omp target map(tofrom:core,thread)
        {
          #pragma omp teams
          {
            int team_num = omp_get_team_num();
            if (team_num == 1) {
              #pragma omp parallel
              {
                #pragma omp master
                {
                  printf("Total GPU threads %d\n", omp_get_num_procs());
                }
                #pragma omp cancel parallel
              }
            }
            printf("Hello from GPU core %d\n", team_num);
          }
        }
      }
    }
  } else {
    int b = 0;
    #pragma omp parallel reduction(+ : b)
    {
      #pragma omp master
      {
        printf("Number of CPU cores %d\n", omp_get_num_procs());
      }

      printf("Hello from CPU thread %d\n", omp_get_thread_num());
      // Workers using their own threads for their tasks
      b += 1;
    }
    printf("Value of b: %d\n", b);
  }

  return 0;
}
