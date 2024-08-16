#include <stdio.h>
#include <stdlib.h>

#define N 1000000000

void vector_add(float *out, float *a, float *b, long n) {
    for (long i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *out;

    // Allocate memory
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    if (a == NULL || b == NULL || out == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize arrays
    for (long i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Perform vector addition
    vector_add(out, a, b, N);

    // Verify the result
    for (long i = 0; i < N; i++) {
        if (out[i] != 3.0f) {
            printf("Error: out[%ld] = %f\n", i, out[i]);
            break;
        }
    }

    // Cleanup
    free(a);
    free(b);
    free(out);

    printf("Vector addition completed successfully\n");

    return 0;
}