CC=nvcc
# CFLAGS=

bitonic: bitonic_sort.cu
	$(CC) bitonic_sort.cu -o build/bitonic_sort
