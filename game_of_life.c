#include <string.h>
#include <unistd.h>
#include <math.h>
#include "utils.h"
#include "mpi_utils.h"
#define MAX_SIZE 400000


void handleParams(int argc, char **argv, int* noSteps) {
  if(argc < 2) {
    printf("You need at least one parameter: Number of steps!");
    exit(1);
  }
  *noSteps = atoi(argv[1]);

  switch(argc) {
    case(3): {
      initGrid(atoi(argv[2]), WIDTH);
      break;
    }
    case(4): {
      initGrid(atoi(argv[2]), atoi(argv[3]));
      break;
    }
    default: {
      initGrid(HEIGHT, WIDTH);
    }
  }
}

int main(int argc, char **argv) {
  int noSteps = NO_STEPS;
  handleParams(argc, argv, &noSteps);
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int slave_limit = sqrt(world_size - 1);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  //Only master does this
  if(world_rank == 0) {
    FILE *in, *out;
    int matrix_size;
    int lines[MAX_SIZE], cols[MAX_SIZE];
    in = fopen("input.dat","r");
    out = fopen("output.dat","w");
    fscanf(in, "%d", &matrix_size);
    int no_pos = 0;
    while(!feof(in)) {
      fscanf(in, "%d", &lines[no_pos]);
      fscanf(in, "%d", &cols[no_pos++]);
    }
    fclose(in);

    int size = matrix_size/slave_limit;
    int slave_matrix[size][size];
    for(int i = 0; i < slave_limit; i++)
      for(int j = 0; j < slave_limit; j++) {
        for(int k = 0; k < size; k++)
          for(int l = 0; l < size; l++)
            slave_matrix[k][l] = 0;
        for(int k = 0; k < no_pos - 1; k++) {
          if(lines[k] >= matrix_size/slave_limit * i && lines[k] <= matrix_size/slave_limit*(i+1) - 1)
            if(cols[k] >= matrix_size/slave_limit * j && cols[k] <= matrix_size/slave_limit*(j+1) - 1)
              slave_matrix[lines[k] -  size * i][cols[k] - size * j] = 1;
        }
        MPI_Send(&size, 1, MPI_INT, i*slave_limit+j + 1, MASTER_MATRIX_SIZE_SEND, MPI_COMM_WORLD);
        MPI_Send(&slave_matrix, size*size, MPI_INT, i*slave_limit+j + 1, MASTER_MATRIX_SEND, MPI_COMM_WORLD);
      }

    fprintf(out, "%d\n", matrix_size);
    MPI_Status status;
    for(int i = 0; i < slave_limit; i++)
      for(int j = 0; j < slave_limit; j++) {
        MPI_Recv(&slave_matrix, size*size, MPI_INT, i*slave_limit+j+1, MASTER_MATRIX_SEND, MPI_COMM_WORLD, &status);
        for(int k = 0; k < size; k++)
          for(int l = 0; l < size; l++)
            if(slave_matrix[k][l])
              fprintf(out, "%d %d ", k + size * i, l + size * j);
      }
    fprintf(out, "\n");
    fclose(out);
  }

  if(world_rank > 0) {
    //Receive size and Matrix from master
    int size;
    MPI_Status status;
    MPI_Recv(&size, 1, MPI_INT, MASTER, MASTER_MATRIX_SIZE_SEND, MPI_COMM_WORLD, &status);

    int matrix[size][size];
    MPI_Recv(&matrix, size*size, MPI_INT, MASTER, MASTER_MATRIX_SEND, MPI_COMM_WORLD, &status);

    //Vectors to be passed between slaves
    int left_vector[size + 2];
    int top_vector[size + 1];
    int bottom_vector[size + 1];
    int right_vector[size];
    int column_left[size];
    int column_right[size];
    int row_top[size];
    int row_bottom[size];
    //Initialize vectors
    for(int i = 0; i < size; i++)
      left_vector[i] = top_vector[i] = bottom_vector[i] = right_vector[i] = 0;
    left_vector[size] = left_vector[size + 1] = top_vector[size] = bottom_vector[size] = 0;
    //Where to pass and receive
    int pass_right = 1;
    int pass_left = 1;
    int pass_top = 1;
    int pass_bottom = 1;
    if(world_rank % slave_limit == 1)
      pass_left = 0;
    if(world_rank % slave_limit == 0)
      pass_right = 0;
    if(world_rank >= 1 && world_rank <= slave_limit)
      pass_top = 0;
    if(world_rank >= slave_limit*(slave_limit - 1) + 1 && world_rank <= slave_limit*slave_limit)
      pass_bottom = 0;
    for(int step = 0; step < noSteps; step++) {
      if(pass_left && pass_top) {
        MPI_Send(&matrix[0][0], 1, MPI_INT, world_rank - slave_limit - 1, CORNER_SEND, MPI_COMM_WORLD);
        MPI_Recv(&left_vector[0], 1, MPI_INT, world_rank - slave_limit - 1, CORNER_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_left && pass_bottom) {
        MPI_Send(&matrix[size - 1][0], 1, MPI_INT, world_rank + slave_limit - 1, CORNER_SEND, MPI_COMM_WORLD);
        MPI_Recv(&left_vector[size + 1], 1, MPI_INT, world_rank + slave_limit - 1, CORNER_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_right && pass_top) {
        MPI_Send(&matrix[0][size - 1], 1, MPI_INT, world_rank - slave_limit + 1, CORNER_SEND, MPI_COMM_WORLD);
        MPI_Recv(&top_vector[size], 1, MPI_INT, world_rank - slave_limit + 1, CORNER_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_right && pass_bottom) {
        MPI_Send(&matrix[size-1][size - 1], 1, MPI_INT, world_rank + slave_limit + 1, CORNER_SEND, MPI_COMM_WORLD);
        MPI_Recv(&bottom_vector[size], 1, MPI_INT, world_rank + slave_limit + 1, CORNER_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_left) {
        for(int i = 0; i < size; i++)
          column_left[i] = matrix[i][0];
        MPI_Send(&column_left, size, MPI_INT, world_rank - 1, VECTOR_SEND, MPI_COMM_WORLD);
        MPI_Recv(&left_vector[1], size, MPI_INT, world_rank - 1, VECTOR_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_right) {
        for(int i = 0; i < size; i++)
          column_right[i] = matrix[i][size - 1];
        MPI_Send(&column_right, size, MPI_INT, world_rank + 1, VECTOR_SEND, MPI_COMM_WORLD);
        MPI_Recv(&right_vector[0], size, MPI_INT, world_rank + 1, VECTOR_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_top) {
        for(int i = 0; i < size; i++)
          row_top[i] = matrix[0][i];
        MPI_Send(&row_top, size, MPI_INT, world_rank - slave_limit, VECTOR_SEND, MPI_COMM_WORLD);
        MPI_Recv(&top_vector[0], size, MPI_INT, world_rank - slave_limit, VECTOR_SEND, MPI_COMM_WORLD, &status);
      }
      if(pass_bottom) {
        for(int i = 0; i < size; i++)
          row_bottom[i] = matrix[size-1][i];
        MPI_Send(&row_bottom, size, MPI_INT, world_rank + slave_limit, VECTOR_SEND, MPI_COMM_WORLD);
        MPI_Recv(&bottom_vector[0], size, MPI_INT, world_rank + slave_limit, VECTOR_SEND, MPI_COMM_WORLD, &status);
      }

      //HERE IS THE PROCESSING!
      if (world_rank == 1) {
      int *h_cells = (int*)calloc(size * size, sizeof(int));

      for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
          h_cells[i * size + j] = matrix[i][j];

      h_cells = newGeneration(h_cells, top_vector, bottom_vector, right_vector,
                              left_vector, size, size);

      for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
          matrix[i][j] = h_cells[i * size + j];
      }
      //END PROCESSING
    }
    MPI_Send(&matrix, size*size, MPI_INT, MASTER, MASTER_MATRIX_SEND, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}
