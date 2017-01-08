#include <string.h>
#include <unistd.h>
#include "utils.h"
#include "mpi_utils.h"

void handleParams(int argc, char **argv) {
  switch(argc) {
    case(2): {
      initGrid(atoi(argv[1]), WIDTH);
      break;
    }
    case(3): {
      initGrid(atoi(argv[1]), atoi(argv[2]));
      break;
    }
    default: {
      initGrid(HEIGHT, WIDTH);
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  printf("Hello from the process no %d\n", world_rank);

  MPI_Finalize();
  // handleParams(argc, argv);
  // setRandomAlive(10);
  // int step;
  // system("clear");
  // for(step = 0; step <= NO_STEPS; step++) {
  //   printf("%d\n", step);
  //   printGrid();
  //   usleep(1000000);
  //   gridStep();
  //   system("clear");
  // }
}
