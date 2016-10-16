build: game_of_life

game_of_life: game_of_life.o utils.o
	g++ game_of_life.o utils.o -o game_of_life

game_of_life.o: game_of_life.cpp
	g++ -c game_of_life.cpp -o game_of_life.o

utils.o: utils.c utils.h
	g++ -c utils.c -o utils.o

.PHONY: clean
clean:
	rm -rf *.o game_of_life
