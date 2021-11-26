#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

#define ARR_TYPE unsigned long
#define ARR_TYPE_MAX LONG_MAX

int askYesNo();
void fillRandom(ARR_TYPE *a, ARR_TYPE nb);
void fillUser(ARR_TYPE *a, ARR_TYPE nb, ARR_TYPE last);
ARR_TYPE checkSorted(ARR_TYPE *a, ARR_TYPE nb);
__device__ void swap(ARR_TYPE *a, ARR_TYPE *b);
__device__ ARR_TYPE intPow(ARR_TYPE base, ARR_TYPE exp);


__global__ void bitonicSort(ARR_TYPE *a, ARR_TYPE nb, int step, int stage) {
    ARR_TYPE index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seqL = intPow(2, step);
    unsigned int N = seqL / intPow(2, stage - 1);
    unsigned int shift = N / 2;
    char working = (index % N) < shift;
    char ascending = (index / seqL) % 2 == 0;

    if(index < nb && working){
    
        if(index + shift > nb && working)
            printf("stp: %d, stg: %d, blckIdx: %d, thrdIdx: %d, idx: %d, shftdIdx: %d, wrk: %d, asc: %d, shft: %d\n", 
                step, stage, blockIdx.x, threadIdx.x, index, index + shift, working, ascending, shift);

        if(ascending){
            if(a[index] > a[index + shift] == 1)
                swap(a + index, a + index + shift);       
        }
        else
            if(a[index] < a[index + shift] == 1)
                swap(a + index, a + index + shift);
    }
}


int main(void) {
    printf("\
 __                __        \n\
|__).|_ _  _ . _  (_  _  _|_ \n\
|__)||_(_)| )|(_  __)(_)| |_ \n\
\n");
    ARR_TYPE *a, *d_a;
    unsigned long nb, size;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    printf("Enter the number of elements of the array (inputs not in range 2^x will be zero padded): ");
    scanf("%lu", &nb);
    getchar(); // Flush newline
    if(nb < 1){
        printf("Error: Cant sort array with less than 1 element");
        return 0;
    }
    
    int exp = ceil(log(nb)/log(2));
    unsigned long newNb = pow(2, exp);
    unsigned long zeros = newNb - nb;
    nb = newNb;
    size = nb * sizeof(ARR_TYPE);
    printf("Number of elements is 2^%d (=%lu) padded with %lu zeros totalling a size of %lu\n", exp, nb, zeros, size);

    a = (ARR_TYPE *)malloc(size); 
    
    printf("Do you want to fill the array by hand? (y\\n): ");
    if(askYesNo())
        fillUser(a, nb, nb - zeros);
    else
        fillRandom(a, nb);
    
    printf("\nArray construction complete. Press enter to begin sort. ");
    getchar(); getchar(); 

    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    int numBlocks = nb / props.maxThreadsPerBlock + 1;
    int numThreads = props.maxThreadsPerBlock;
    for(int step=1; step <= exp; step++)
        for(int stage=1; stage <= step; stage++)
            bitonicSort<<<numBlocks, numThreads>>>(d_a, nb, step, stage);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    printf("Do you want to print the results? (y\\n): ");
    if(askYesNo()){
        printf("\n\nIndex\t\tValue\n");
        for(ARR_TYPE i=0; i < nb; i++)
            printf("%lu\t\t%d\n", i, a[i]);
    }

    printf("\nSorted %lu elements using %d blocks and %d threads with %lu errors.\n", nb, numBlocks, numThreads, checkSorted(a, nb));

    free(a);
    cudaFree(d_a);

    printf("Press enter to terminate. ");
    getchar();

    return 0;
}

int askYesNo(){
    int answer = getchar();
    return answer == 121 || answer == 89; // Check if answer is y or Y
}

void fillRandom(ARR_TYPE *a, ARR_TYPE nb){
    srand(time(NULL));
    for(ARR_TYPE i=0; i < nb; i++)
        a[i] = (ARR_TYPE) ((double) rand() / RAND_MAX * ARR_TYPE_MAX);
}

void fillUser(ARR_TYPE *a, ARR_TYPE nb, ARR_TYPE last){
    for(ARR_TYPE i=0; i < nb; i++){
        if(i < last){
            printf("Please input a number for the %dth placement: ", i);
            scanf("%d", a + i);
        }
        else
            a[i] = 0;
    }
}

ARR_TYPE checkSorted(ARR_TYPE *a, ARR_TYPE nb){
    if(nb == 0)
        return 1;

    ARR_TYPE count = 0;

    for(ARR_TYPE i=1; i < nb; i++)
        if(a[i-1] > a[i])
            count++;

    return count;
}

 __device__ void swap(ARR_TYPE *a, ARR_TYPE *b){
    ARR_TYPE temp = *a;
    *a = *b;
    *b = temp;
 }

__device__ ARR_TYPE intPow(ARR_TYPE base, ARR_TYPE exp)
{
    ARR_TYPE result = 1;
    while (exp){
        if (exp % 2)
           result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}
