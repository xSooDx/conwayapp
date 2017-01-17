#include <cstdio>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

using namespace std;

__global__ void cellsKernel(char *cells, int height, int width, char *resultCells,
							char *borderTop, char *borderRight, char *bordertBot, char *borderLeft)
{
	int worldSize = height * width;
	int currentCellX, currentCellY, aliveCells, currentRow;
	int topOrBot;
	int leftOrRight;

	int N, NE, E, SE, S, SW, W, NW;

	for (int cellId = blockIdx.x * blockDim.x + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x) {
		currentCellY = cellId % width; // cell's index from the **matrix**
		currentCellX = cellId - currentCellY; // the number of cells in the **matrix** until the current one
		currentRow = cellId / width;

		aliveCells = 0;
		topOrBot = 0;
		leftOrRight = 0;

		N = (currentRow == 0) ? borderTop[currentCellY] : cells[currentCellX - width + currentCellY];
		S = (currentRow + 1 == height) ? bordertBot[currentCellY] : cells[currentCellX + width + currentCellY];
		W = (currentCellY == 0) ? borderLeft[currentRow + 1] : cells[currentCellX + currentCellY - 1];
		E = (currentCellY + 1 == width) ? borderRight[currentRow] : cells[currentCellX + currentCellY + 1];

		if (currentRow == 0)
			NE = borderTop[currentCellY + 1];
		else if (currentCellY + 1 == width)
			NE = borderRight[currentRow - 1];
		else
			NE = cells[currentCellX - width + currentCellY + 1];
		if (currentCellY == 0)
			NW = borderLeft[currentCellY];
		else if (currentRow == 0)
			NW = borderTop[currentCellY - 1];
		else
			NW = cells[currentCellX - width + currentCellY - 1];
		if (currentRow + 1 == height)
			SE = bordertBot[currentCellY + 1];
		else if (currentCellY + 1 == width)
			SE = borderRight[currentRow + 1];
		else
			SE = cells[currentCellX + width + currentCellY + 1];
		if (currentCellY == 0)
			SW = borderLeft[currentRow + 2];
		else if (currentRow + 1 == height)
			SW = bordertBot[currentCellY - 1];
		else
			SW = cells[currentCellX + width + currentCellY - 1];

		aliveCells = N + S + E + W + NE + SE + SW + NW;

		resultCells[currentCellX + currentCellY] = (aliveCells == 3 || (aliveCells == 2 && cells[currentCellX + currentCellY] == 1)) ? 1 : 0;
	}
}

void computeCells(char *&cells, int height, int width, char *&resultCells, int threadsCount,
									char *borderTop, char *borderRight, char *borderBot, char *borderLeft)
{
	if ((width * height) % threadsCount != 0) {
		fprintf(stderr, "%s", "The product of square dimensions must be multiple of the number of threads!\n");
		printf("%s", "The product of square dimensions must be multiple of the number of threads!\n");
		exit(1);
	}

	int blocksCount = min(32768, (height * width) / threadsCount);

	cellsKernel <<<blocksCount, threadsCount >>> (cells, height, width, resultCells, borderTop, borderRight, borderBot, borderLeft);
}

int* newGeneration(int *h_cells, int *h_borderTop, int *h_borderBot,
									 int *h_borderRight, int *h_borderLeft, int height, int width)
{
	char *d_cells, *d_resultCells, *d_borderTop, *d_borderRight, *d_borderBot, *d_borderLeft;

	int sizeWorld = height * width;

	size_t cudaStatus;

	cudaStatus = cudaMalloc(&d_cells, sizeWorld * sizeof(int));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_resultCells, sizeWorld * sizeof(int));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderTop, (width + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderRight, (height) * sizeof(int));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderBot, (width + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderLeft, (height + 2) * sizeof(int));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_cells, h_cells, sizeWorld * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderTop, h_borderTop, width + 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderBot, h_borderBot, width + 1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderRight, h_borderRight, height, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderLeft, h_borderLeft, height + 2, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	computeCells(d_cells, height, width, d_resultCells, worldSize, d_borderTop,
							 d_borderRight, d_borderBot, d_borderLeft);

	cudaStatus = cudaMemcpy(h_cells, d_resultCells, worldSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaFree(d_cells);
	cudaFree(d_resultCells);
	cudaFree(d_borderBot);
	cudaFree(d_borderLeft);
	cudaFree(d_borderRight);
	cudaFree(d_borderTop);

	return h_cells;
}
