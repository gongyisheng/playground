#include <stdio.h>
#include <stdlib.h>

#define N 1<<20
#define BLOCK_SIZE 256

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i ++){
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    // Note: The `cudaMalloc` function is defined to take a `void**` as its first argument. 
    // This is because `cudaMalloc` can allocate memory for any data type, not just float. 
    // Using `void**` makes the function more generic.
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Launch kernel
    vector_add<<<1, 1>>>(d_out, d_a, d_b, N);

    // Transfer result back to host
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (out[i] != 3.0f) {
            printf("Error: out[%ld] = %f\n", i, out[i]);
            break;
        }
    }

    // Cleanup
    free(a); 
    free(b); 
    free(out);
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_out);

    printf("Vector addition completed successfully\n");

    return 0;
}