#include "utils.h"

//This function initializes the grid
void initGrid(int h, int w) {
  *&width = w;
  *&height = h;
  int i;
  grid = (int**)calloc(h, sizeof(int*));
  for(i = 0; i < h; i++)
    grid[i] = (int*)calloc(w, sizeof(int));

  old_grid = (int**)calloc(h, sizeof(int*));
  for(i = 0; i < h; i++)
    old_grid[i] = (int*)calloc(w, sizeof(int));
}

void copyGrid() {
  int i, j;
  for(i = 0; i < height; i++)
    for(j = 0; j < height; j++)
      old_grid[i][j] = grid[i][j];
}

//This function shows the gread
void printGrid(){
  int i, j;
  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++)
      switch(grid[i][j]) {
        case(DEAD): {printf(". "); break;}
        case(ALIVE): {printf("* "); break;}
      }
    printf("\n");
  }
}

//Function that gets how many alive cells are nearby
int getAlive(int i, int j) {
  int totalAlive = 0;
  int k, m;
  for(k = i-1; k <= i+1; k++)
    for(m = j-1; m <= j+1; m++)
      if(k >= 0 && k <= height-1 && m >= 0 && m <= width-1)
        totalAlive += old_grid[k][m];
  return totalAlive;
}

//Function that generates random life
void setRandomAlive(int x) {
  grid[5][4] = ALIVE;
  grid[5][3] = ALIVE;
  grid[5][5] = ALIVE;
  // srand(time(NULL));
  // while(x > 0) {
  //   int rand_i = rand() % height;
  //   int rand_j = rand() % width;
  //   grid[rand_i][rand_j] = ALIVE;
  //   x--;
  // }
}

//Function that generates a step for each cell
void gridStep() {
  copyGrid();
  int i, j;
  for(i = 0; i < height; i++)
    for(j = 0; j < width; j++) {
      int no_cells = getAlive(i, j);
      if(grid[i][j] == ALIVE && no_cells < 2)
        grid[i][j] = DEAD;
      if(grid[i][i] == ALIVE && (no_cells == 2 || no_cells == 3))
        grid[i][j] = ALIVE;
      if(grid[i][j] == ALIVE && no_cells > 3)
        grid[i][j] = DEAD;
      if(grid[i][j] == DEAD && no_cells == 3)
        grid[i][j] = ALIVE;
    }
}
