printf '\n add.cpp \n'
g++ -O3 add.cpp -o add_cpp
./add_cpp

printf '\n add1.cu \n'
nvcc add1.cu -o add1
./add1

printf '\n add2wrong.cu \n'
nvcc add2wrong.cu -o add2wrong
./add2wrong

printf '\n add3if.cu \n'
nvcc add3if.cu -o add3if
./add3if

printf '\n hello4.cu \n'
nvcc add4device.cu -o add4device
./add4device

