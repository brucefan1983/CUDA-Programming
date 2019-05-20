all: ljmd

CC = g++
CFLAGS = -O3

ljmd: initialize.o integrate.o neighbor.o \
	force.o memory.o main.o
	$(CC) -o ljmd \
	initialize.o integrate.o neighbor.o \
	force.o memory.o main.o

initialize.o: initialize.cpp
	$(CC) $(CFLAGS) -c initialize.cpp
integrate.o: integrate.cpp
	$(CC) $(CFLAGS) -c integrate.cpp
memory.o: memory.cpp
	$(CC) $(CFLAGS) -c memory.cpp
neighbor.o: neighbor.cpp
	$(CC) $(CFLAGS) -c neighbor.cpp
force.o: force.cpp
	$(CC) $(CFLAGS) -c force.cpp
main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm *o ljmd
