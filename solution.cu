#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void fillRandom(int *a, int nb){
    for(int i=0; i < nb; i++){
        a[i] = rand() % 100000000;
    }
}

void fillUser(int *a, int nb, int last){
    for(int i=0; i < nb; i++){
        if(i < last){
            printf("Please input a number for the %dth placement: ", i);
            scanf("%d", a + i);
        }
        else
            a[i] = 0;
    }
}

int checkSorted(int *a, int nb){
	if(nb == 0){
		return 1;
	}

    for(int i=1; i < nb; i++){
        if(a[i-1] > a[i]){
        	return 0;
        }
    }

    return 1;
}

 __device__ void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
 }

__global__ void bitonicSort(int *a, unsigned long nb, int step, int stage) {
	// int index = blockIdx.x * 1024 + threadIdx.x;
	int index = threadIdx.x;
	// printf("blockIdx: %d, threadIdx: %d, index: %d\n", blockIdx.x, threadIdx.x, index);

    unsigned int seqL = pow(2, step);
    unsigned int N = seqL / pow(2, stage - 1);
    unsigned int shift = N / 2;
    short working = index % N < shift;
    short ascending = index / seqL % 2 == 0;
    
    if(working)
        // printf("ThreadIdx: %d\tAscending: %d\n", index, ascending);
        if(ascending){
            if(a[index] > a[index + shift] == 1)
                swap(a + index, a + index + shift);       
        }
        else
            if(a[index] < a[index + shift] == 1)
                swap(a + index, a + index + shift);
}


int main(void) {
    srand(time(NULL));

    int *a, *d_a;
    unsigned long nb, size;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf( "Multiproc: %d\n", prop.multiProcessorCount );
	printf( "Shared mem: %d\n",prop.sharedMemPerBlock );
	printf( "Registers per mp: %d\n", prop.regsPerBlock );
	printf( "Threads in warp: %d\n", prop.warpSize );
	printf( "Threads: %d\n", prop.maxThreadsPerBlock );
	printf( "Max Thread dimensions: (%d,%d,%d)\n",
	prop.maxThreadsDim[0],
	prop.maxThreadsDim[1],
	prop.maxThreadsDim[2] );
	printf( "Max grid dimensions: (%d, %d, %d)\n",
	prop.maxGridSize[0], prop.maxGridSize[1],
	prop.maxGridSize[2] );

    printf("Enter the number of elements of the array (inputs not in range 2^x will be zero padded): ");
    scanf("%lu", &nb);
    
    int exp = ceil(log(nb)/log(2));
    unsigned long newNb = pow(2, exp);
    unsigned long zeros = newNb - nb;
    nb = newNb;
    size = nb * sizeof(int);
    printf("Number of elements is 2^%d (=%lu) padded with %lu zeros totalling a size of %lu\n", exp, nb, zeros, size);

    a = (int *)malloc(size); 
    
    printf("Do you want to fill the array by hand? (y\\n): ");
    getchar(); // Flush
    int answer = getchar();
    answer = answer == 121 || answer == 89; // Check if answer is y or Y

    if(answer)
        fillUser(a, nb, nb - zeros);
    else
        fillRandom(a, nb); 

    // fillUser(a, nb, nb - zeros);
    
    printf("\nArray construction complete. Press enter to begin sort. ");
    getchar(); getchar(); 

    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);    

    for(int step=1; step <= exp; step++)
        for(int stage=1; stage <= step; stage++)
        	// bitonicSort<<<1, nb>>>(d_a, nb, step, stage);
            bitonicSort<<<nb / 1024, 1024>>>(d_a, nb, step, stage);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("\n\nIndex\t\tValue\n");
    for(unsigned long i=0; i < nb; i++)
        printf("%lu\t\t%d\n", i, a[i]);

    // Check sorted
    if(checkSorted(a, nb))
    	printf("Sorted");
    else
    	printf("Unsorted");

    // Cleanup
    free(a);
    cudaFree(d_a);

    return 0;
}