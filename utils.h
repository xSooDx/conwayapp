#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ALIVE 1
#define DEAD 0

static const int WIDTH = 10;
static const int HEIGHT = 10;
static const int NO_STEPS = 100;
static int width;
static int height;
static int **grid;
static int **old_grid;

//Used for slave matrixes initialization
int **initializedMatrix(int);

//This function initializes the grid
void initGrid(int, int);

//This function shows the grid
void printGrid();

//Check adjacent life
int getAlive(int, int);

//Function that randomly brings cells to life
void setRandomAlive(int);

int* newGeneration(int *h_cells, int *h_borderTop, int *h_borderBot,
                   int *h_borderRight, int *h_borderLeft, int height, int width);

//Function that generates a step for each cell
void gridStep();
#endif
