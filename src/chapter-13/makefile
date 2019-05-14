all: ljmd

CC = g++
CFLAGS = -O3

ljmd: initialize.o integrate.o neighbor.o force.o main.o
	$(CC) -o ljmd \
	initialize.o integrate.o neighbor.o force.o main.o

initialize.o: initialize.cpp
	$(CC) $(CFLAGS) -c initialize.cpp
integrate.o: integrate.cpp
	$(CC) $(CFLAGS) -c integrate.cpp
neighbor.o: neighbor.cpp
	$(CC) $(CFLAGS) -c neighbor.cpp
force.o: force.cpp
	$(CC) $(CFLAGS) -c force.cpp
main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm *o ljmd
