build: game_of_life

game_of_life: game_of_life.o utils.o mpi_utils.o kernel.o
	mpicc game_of_life.o utils.o mpi_utils.o kernel.o -o game_of_life -lm -L/usr/local/cuda/lib64/ -lcudart -lstdc++ 

game_of_life.o: game_of_life.c
	mpicc -c game_of_life.c -o game_of_life.o

kernel.o: kernel.cu
	nvcc -c kernel.cu -o kernel.o

utils.o: utils.c utils.h
	mpicc -c utils.c -o utils.o

mpi_utils.o: mpi_utils.c mpi_utils.h
	mpicc -c mpi_utils.c -o mpi_utils.o

.PHONY: clean
clean:
	rm -rf *.o game_of_life
