all: ljmd

CC = nvcc
CFLAGS = -O3 -arch=sm_75

ljmd: initialize.o integrate.o neighbor.o force.o memory.o reduce.o main.o
	$(CC) -o ljmd \
	initialize.o integrate.o neighbor.o force.o memory.o reduce.o main.o

initialize.o: initialize.cu
	$(CC) $(CFLAGS) -c initialize.cu
integrate.o: integrate.cu
	$(CC) $(CFLAGS) -c integrate.cu
memory.o: memory.cu
	$(CC) $(CFLAGS) -c memory.cu
neighbor.o: neighbor.cu
	$(CC) $(CFLAGS) -c neighbor.cu
force.o: force.cu
	$(CC) $(CFLAGS) -c force.cu
reduce.o: reduce.cu
	$(CC) $(CFLAGS) -c reduce.cu
main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu

clean:
	rm *o ljmd
	
