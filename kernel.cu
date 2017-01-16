#include <cstdio>
#include <iostream>
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
		currentCellY = cellId % width; // indexul celulei de pe randul curent din **matrice**
		currentCellX = cellId - currentCellY; // numarul total de celule pana la randul curent din **matrice**
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

int main()
{
	char *d_cells, *d_resultCells, *d_borderTop, *d_borderRight, *d_borderBot, *d_borderLeft;
	char *h_cells = (char*)calloc(100, sizeof(char));

	char *h_borderTop = (char*)calloc(11, sizeof(char));
	char *h_borderBot = (char*)calloc(11, sizeof(char));
	char *h_borderRight = (char*)calloc(10, sizeof(char));
	char *h_borderLeft = (char*)calloc(12, sizeof(char));

	size_t cudaStatus;

	h_cells[0] = 1;
	
	h_borderTop[0] = 1;
	h_borderLeft[0] = 1;
	h_borderLeft[1] = 1;

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++)
			printf("%d ", h_cells[i * 10 + j]);
		cout << endl;
	}

	cudaStatus = cudaMalloc(&d_cells, 100 * sizeof(char));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_resultCells, 100 * sizeof(char));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderTop, 100 * sizeof(char));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderRight, 100 * sizeof(char));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderBot, 100 * sizeof(char));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMalloc(&d_borderLeft, 100 * sizeof(char));
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_cells, h_cells, 100 * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderTop, h_borderTop, 11, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderBot, h_borderBot, 11, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderRight, h_borderRight, 11, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	cudaStatus = cudaMemcpy(d_borderLeft, h_borderLeft, 12, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		return 1;

	computeCells(d_cells, 10, 10, d_resultCells, 100, d_borderTop, d_borderRight, d_borderBot, d_borderLeft);
	
	char *h = (char*)calloc(100, sizeof(char));
	cudaStatus = cudaMemcpy(h, d_resultCells, 100 * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		return 1;

	printf("CUDA Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++)
			printf("%d ", h[i * 10 + j]);
		cout << endl;
	}

	system("pause");

	cudaFree(d_cells);
	cudaFree(d_resultCells);
	cudaFree(d_borderBot);
	cudaFree(d_borderLeft);
	cudaFree(d_borderRight);
	cudaFree(d_borderTop);

	return 0;
}