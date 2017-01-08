build: game_of_life

game_of_life: game_of_life.o utils.o mpi_utils.o
	mpic++ game_of_life.o utils.o mpi_utils.o -o game_of_life

game_of_life.o: game_of_life.cpp
	mpic++ -c game_of_life.cpp -o game_of_life.o

utils.o: utils.cpp utils.h
	mpic++ -c utils.cpp -o utils.o

mpi_utils.o: mpi_utils.cpp mpi_utils.h
	mpic++ -c mpi_utils.cpp -o mpi_utils.o

.PHONY: clean
clean:
	rm -rf *.o game_of_life
