printf '\n global.cu \n'
nvcc global.cu -o global
./global

printf '\n device_query.cu \n'
nvcc device_query.cu -o device_query
./device_query

