build: game_of_life

game_of_life: game_of_life.o utils.o
	gcc game_of_life.o utils.o -o game_of_life

game_of_life.o: game_of_life.c
	gcc -c game_of_life.c -o game_of_life.o

utils.o: utils.c utils.h
	gcc -c utils.c -o utils.o

.PHONY: clean
clean:
	rm -rf *.o game_of_life
