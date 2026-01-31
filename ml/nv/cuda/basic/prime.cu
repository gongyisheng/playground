#include "utils.h"
#include <chrono>

/*
 * Prime Number Finder using CUDA - Performance Optimized
 *
 * Version 1: Baseline - One kernel per candidate (slow)
 * Version 2: Batch processing with dependency handling
 * Version 3: Basic Sieve of Eratosthenes
 * Version 4: Optimized Sieve with wheel factorization
 * Version 5: Ultra-optimized with shared memory + coalescing
 *
 * Compile: nvcc -O3 -arch=native -o build/prime basic/prime.cu
 * Run: ./build/prime [count] [version]
 */

// nvcc -O3 -o build/prime basic/prime.cu && ./build/prime 100000

//=============================================================================
// VERSION 1: BASELINE - One kernel per candidate
//=============================================================================
__global__ void check_divisibility_v1(int candidate, int *primes, int prime_count, int *is_composite) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < prime_count) {
        int prime = primes[idx];
        if (prime * prime <= candidate && candidate % prime == 0) {
            *is_composite = 1;
        }
    }
}

void find_primes_v1(int max_primes, int *h_primes, bool verbose) {
    int *d_primes, *d_is_composite;
    CUDA_ERROR_CHECK(cudaMalloc(&d_primes, max_primes * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_is_composite, sizeof(int)));

    h_primes[0] = 2;
    int prime_count = 1;
    CUDA_ERROR_CHECK(cudaMemcpy(d_primes, h_primes, sizeof(int), cudaMemcpyHostToDevice));

    int candidate = 3;
    int block_size = 256;

    while (prime_count < max_primes) {
        CUDA_ERROR_CHECK(cudaMemset(d_is_composite, 0, sizeof(int)));

        int num_blocks = (prime_count + block_size - 1) / block_size;
        check_divisibility_v1<<<num_blocks, block_size>>>(candidate, d_primes, prime_count, d_is_composite);

        int is_composite;
        CUDA_ERROR_CHECK(cudaMemcpy(&is_composite, d_is_composite, sizeof(int), cudaMemcpyDeviceToHost));

        if (!is_composite) {
            h_primes[prime_count] = candidate;
            CUDA_ERROR_CHECK(cudaMemcpy(d_primes + prime_count, &candidate, sizeof(int), cudaMemcpyHostToDevice));
            prime_count++;
        }
        candidate += 2;
    }

    CUDA_ERROR_CHECK(cudaFree(d_primes));
    CUDA_ERROR_CHECK(cudaFree(d_is_composite));
}

//=============================================================================
// VERSION 2: BATCH - Process candidates in batches, with correct handling
//=============================================================================
#define BATCH_SIZE 4096

__global__ void check_batch_v2(int *candidates, int num_candidates, int *primes, int prime_count, int *results) {
    int cand_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (cand_idx >= num_candidates) return;

    int candidate = candidates[cand_idx];
    __shared__ int is_composite;

    if (tid == 0) is_composite = 0;
    __syncthreads();

    // Check against known primes
    for (int i = tid; i < prime_count && !is_composite; i += blockDim.x) {
        int prime = primes[i];
        if ((long long)prime * prime > candidate) break;
        if (candidate % prime == 0) {
            atomicExch(&is_composite, 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        results[cand_idx] = is_composite;
    }
}

void find_primes_v2(int max_primes, int *h_primes, bool verbose) {
    int *d_primes, *d_candidates, *d_results;
    int *h_candidates = (int *)malloc(BATCH_SIZE * sizeof(int));
    int *h_results = (int *)malloc(BATCH_SIZE * sizeof(int));

    CUDA_ERROR_CHECK(cudaMalloc(&d_primes, max_primes * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_candidates, BATCH_SIZE * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_results, BATCH_SIZE * sizeof(int)));

    h_primes[0] = 2;
    h_primes[1] = 3;
    int prime_count = 2;
    CUDA_ERROR_CHECK(cudaMemcpy(d_primes, h_primes, 2 * sizeof(int), cudaMemcpyHostToDevice));

    int candidate = 5;
    int block_size = 128;

    while (prime_count < max_primes) {
        // Fill batch with odd candidates not divisible by 3
        int batch_count = 0;
        while (batch_count < BATCH_SIZE) {
            if (candidate % 3 != 0) {
                h_candidates[batch_count++] = candidate;
            }
            candidate += 2;
        }

        CUDA_ERROR_CHECK(cudaMemcpy(d_candidates, h_candidates, batch_count * sizeof(int), cudaMemcpyHostToDevice));

        check_batch_v2<<<batch_count, block_size>>>(d_candidates, batch_count, d_primes, prime_count, d_results);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        CUDA_ERROR_CHECK(cudaMemcpy(h_results, d_results, batch_count * sizeof(int), cudaMemcpyDeviceToHost));

        // Collect new primes
        int new_primes_start = prime_count;
        for (int i = 0; i < batch_count && prime_count < max_primes; i++) {
            if (!h_results[i]) {
                // Verify against newly found primes in this batch
                int cand = h_candidates[i];
                bool is_prime = true;
                for (int j = new_primes_start; j < prime_count; j++) {
                    if ((long long)h_primes[j] * h_primes[j] > cand) break;
                    if (cand % h_primes[j] == 0) {
                        is_prime = false;
                        break;
                    }
                }
                if (is_prime) {
                    h_primes[prime_count++] = cand;
                }
            }
        }

        CUDA_ERROR_CHECK(cudaMemcpy(d_primes, h_primes, prime_count * sizeof(int), cudaMemcpyHostToDevice));
    }

    free(h_candidates);
    free(h_results);
    CUDA_ERROR_CHECK(cudaFree(d_primes));
    CUDA_ERROR_CHECK(cudaFree(d_candidates));
    CUDA_ERROR_CHECK(cudaFree(d_results));
}

//=============================================================================
// VERSION 3: BASIC SIEVE
//=============================================================================
__global__ void sieve_init(bool *is_prime, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= limit) {
        is_prime[idx] = (idx >= 2);
    }
}

__global__ void sieve_mark(bool *is_prime, int limit, int prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = prime * prime + idx * prime;
    if (num <= limit) {
        is_prime[num] = false;
    }
}

void find_primes_v3(int max_primes, int *h_primes, bool verbose) {
    int limit;
    if (max_primes < 6) {
        limit = 15;
    } else {
        double n = (double)max_primes;
        limit = (int)(n * (log(n) + log(log(n)))) + 100;
    }

    bool *d_is_prime;
    CUDA_ERROR_CHECK(cudaMalloc(&d_is_prime, (limit + 1) * sizeof(bool)));

    int block_size = 256;
    int num_blocks = (limit + 1 + block_size - 1) / block_size;

    sieve_init<<<num_blocks, block_size>>>(d_is_prime, limit);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    bool *h_is_prime = (bool *)malloc((limit + 1) * sizeof(bool));

    int sqrt_limit = (int)sqrt((double)limit) + 1;
    for (int p = 2; p <= sqrt_limit; p++) {
        CUDA_ERROR_CHECK(cudaMemcpy(&h_is_prime[p], &d_is_prime[p], sizeof(bool), cudaMemcpyDeviceToHost));
        if (h_is_prime[p]) {
            int count = (limit - p * p) / p + 1;
            int mark_blocks = (count + block_size - 1) / block_size;
            sieve_mark<<<mark_blocks, block_size>>>(d_is_prime, limit, p);
        }
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CUDA_ERROR_CHECK(cudaMemcpy(h_is_prime, d_is_prime, (limit + 1) * sizeof(bool), cudaMemcpyDeviceToHost));

    int prime_count = 0;
    for (int i = 2; i <= limit && prime_count < max_primes; i++) {
        if (h_is_prime[i]) {
            h_primes[prime_count++] = i;
        }
    }

    free(h_is_prime);
    CUDA_ERROR_CHECK(cudaFree(d_is_prime));
}

//=============================================================================
// VERSION 4: OPTIMIZED SIEVE - Odd numbers only, better memory
//=============================================================================
// Mark composite: only store odd numbers, index i represents number 2*i+1
__global__ void sieve_mark_odd(unsigned char *sieve, int sieve_size, int prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Start from prime*prime, step by 2*prime (skip even multiples)
    long long start = (long long)prime * prime;
    long long num = start + (long long)idx * 2 * prime;

    // Convert number to sieve index: num = 2*i+1 => i = (num-1)/2
    if (num > 2 && (num & 1)) {  // odd number
        int sieve_idx = (num - 1) / 2;
        if (sieve_idx < sieve_size) {
            sieve[sieve_idx] = 0;
        }
    }
}

void find_primes_v4(int max_primes, int *h_primes, bool verbose) {
    int limit;
    if (max_primes < 6) {
        limit = 15;
    } else {
        double n = (double)max_primes;
        limit = (int)(n * (log(n) + log(log(n)))) + 100;
    }

    // Only store odd numbers: sieve[i] represents 2*i+1
    int sieve_size = (limit + 1) / 2;

    unsigned char *d_sieve, *h_sieve;
    h_sieve = (unsigned char *)malloc(sieve_size);
    CUDA_ERROR_CHECK(cudaMalloc(&d_sieve, sieve_size));

    // Initialize all as prime
    CUDA_ERROR_CHECK(cudaMemset(d_sieve, 1, sieve_size));

    int block_size = 256;
    int sqrt_limit = (int)sqrt((double)limit) + 1;

    // CPU sieve for small primes
    memset(h_sieve, 1, sieve_size);
    for (int p = 3; p <= sqrt_limit; p += 2) {
        int idx = (p - 1) / 2;
        if (h_sieve[idx]) {
            // Mark on GPU
            long long start = (long long)p * p;
            int count = (limit - start) / (2 * p) + 1;
            if (count > 0) {
                int mark_blocks = (count + block_size - 1) / block_size;
                sieve_mark_odd<<<mark_blocks, block_size>>>(d_sieve, sieve_size, p);
            }
            // Mark on CPU sieve for next iterations
            for (long long m = (long long)p * p; m <= sqrt_limit; m += 2 * p) {
                h_sieve[(m - 1) / 2] = 0;
            }
        }
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CUDA_ERROR_CHECK(cudaMemcpy(h_sieve, d_sieve, sieve_size, cudaMemcpyDeviceToHost));

    // Collect primes
    int prime_count = 0;
    h_primes[prime_count++] = 2;  // Add 2 manually

    for (int i = 1; i < sieve_size && prime_count < max_primes; i++) {
        if (h_sieve[i]) {
            h_primes[prime_count++] = 2 * i + 1;
        }
    }

    free(h_sieve);
    CUDA_ERROR_CHECK(cudaFree(d_sieve));
}

//=============================================================================
// VERSION 5: ULTRA-OPTIMIZED - Bit packing + parallel marking + streams
//=============================================================================
#define BITS_PER_WORD 32

// Each thread marks multiple composites for one prime
__global__ void sieve_mark_v5(unsigned int *sieve, int sieve_bits, int prime, int prime_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Start from prime*prime for this prime
    long long start = (long long)prime * prime;
    // Step by 2*prime (skip even multiples since we only store odd numbers)
    long long step = 2LL * prime;
    long long num = start + (long long)tid * step;

    while (num <= (long long)sieve_bits * 2 + 1) {
        // num = 2*bit_idx + 1 => bit_idx = (num-1)/2
        int bit_idx = (num - 1) / 2;
        if (bit_idx < sieve_bits) {
            int word_idx = bit_idx / BITS_PER_WORD;
            int bit_pos = bit_idx % BITS_PER_WORD;
            atomicAnd(&sieve[word_idx], ~(1u << bit_pos));
        }
        num += step * gridDim.x * blockDim.x;
    }
}

// Initialize sieve with all bits set (all odd numbers are potentially prime)
__global__ void sieve_init_v5(unsigned int *sieve, int num_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_words) {
        sieve[idx] = 0xFFFFFFFF;
    }
}

// Count primes using parallel reduction
__global__ void count_primes_v5(unsigned int *sieve, int num_words, int *count) {
    __shared__ int local_count[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    local_count[tid] = 0;
    if (idx < num_words) {
        local_count[tid] = __popc(sieve[idx]);
    }
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_count[tid] += local_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, local_count[0]);
    }
}

void find_primes_v5(int max_primes, int *h_primes, bool verbose) {
    int limit;
    if (max_primes < 6) {
        limit = 15;
    } else {
        double n = (double)max_primes;
        limit = (int)(n * (log(n) + log(log(n)))) + 100;
    }

    // Bit-packed sieve for odd numbers only
    int sieve_bits = (limit + 1) / 2;  // Number of odd numbers
    int num_words = (sieve_bits + BITS_PER_WORD - 1) / BITS_PER_WORD;

    unsigned int *d_sieve;
    CUDA_ERROR_CHECK(cudaMalloc(&d_sieve, num_words * sizeof(unsigned int)));

    int block_size = 256;
    int init_blocks = (num_words + block_size - 1) / block_size;

    // Initialize
    sieve_init_v5<<<init_blocks, block_size>>>(d_sieve, num_words);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Mark bit 0 (represents 1) as not prime
    unsigned int first_word;
    CUDA_ERROR_CHECK(cudaMemcpy(&first_word, d_sieve, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    first_word &= ~1u;  // Clear bit 0
    CUDA_ERROR_CHECK(cudaMemcpy(d_sieve, &first_word, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // CPU sieve for small primes (up to sqrt(limit))
    int sqrt_limit = (int)sqrt((double)limit) + 1;
    int small_sieve_size = (sqrt_limit + 1) / 2;
    unsigned char *small_sieve = (unsigned char *)malloc(small_sieve_size);
    memset(small_sieve, 1, small_sieve_size);
    small_sieve[0] = 0;  // 1 is not prime

    for (int i = 1; i < small_sieve_size; i++) {
        int p = 2 * i + 1;
        if (p * p > sqrt_limit) break;
        if (small_sieve[i]) {
            // Mark multiples in small sieve
            for (int j = (p * p - 1) / 2; j < small_sieve_size; j += p) {
                small_sieve[j] = 0;
            }
        }
    }

    // Mark composites on GPU for each small prime
    for (int i = 1; i < small_sieve_size; i++) {
        if (small_sieve[i]) {
            int p = 2 * i + 1;
            if ((long long)p * p > limit) break;

            long long start = (long long)p * p;
            long long count = (limit - start) / (2 * p) + 1;
            int mark_blocks = min((int)((count + block_size - 1) / block_size), 1024);

            sieve_mark_v5<<<mark_blocks, block_size>>>(d_sieve, sieve_bits, p, i);
        }
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy back and collect primes
    unsigned int *h_sieve = (unsigned int *)malloc(num_words * sizeof(unsigned int));
    CUDA_ERROR_CHECK(cudaMemcpy(h_sieve, d_sieve, num_words * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int prime_count = 0;
    h_primes[prime_count++] = 2;  // Add 2 manually

    for (int word = 0; word < num_words && prime_count < max_primes; word++) {
        unsigned int w = h_sieve[word];
        while (w && prime_count < max_primes) {
            int bit = __builtin_ctz(w);  // Find lowest set bit
            int idx = word * BITS_PER_WORD + bit;
            int num = 2 * idx + 1;
            if (num <= limit) {
                h_primes[prime_count++] = num;
            }
            w &= w - 1;  // Clear lowest set bit
        }
    }

    free(small_sieve);
    free(h_sieve);
    CUDA_ERROR_CHECK(cudaFree(d_sieve));
}

//=============================================================================
// VERSION 6: MAXIMUM PERFORMANCE - Cooperative groups + better parallelism
//=============================================================================
// Mark all composites for multiple primes in one kernel
__global__ void sieve_mark_multi(unsigned int *sieve, int sieve_bits,
                                  int *primes, int num_primes, int start_prime_idx) {
    int prime_idx = start_prime_idx + blockIdx.y;
    if (prime_idx >= num_primes) return;

    int prime = primes[prime_idx];
    long long start = (long long)prime * prime;
    long long step = 2LL * prime;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long num = start + (long long)tid * step;

    while (num <= (long long)sieve_bits * 2 + 1) {
        int bit_idx = (num - 1) / 2;
        if (bit_idx >= 0 && bit_idx < sieve_bits) {
            int word_idx = bit_idx / BITS_PER_WORD;
            int bit_pos = bit_idx % BITS_PER_WORD;
            atomicAnd(&sieve[word_idx], ~(1u << bit_pos));
        }
        num += step * gridDim.x * blockDim.x;
    }
}

void find_primes_v6(int max_primes, int *h_primes, bool verbose) {
    int limit;
    if (max_primes < 6) {
        limit = 15;
    } else {
        double n = (double)max_primes;
        limit = (int)(n * (log(n) + log(log(n)))) + 100;
    }

    int sieve_bits = (limit + 1) / 2;
    int num_words = (sieve_bits + BITS_PER_WORD - 1) / BITS_PER_WORD;

    unsigned int *d_sieve;
    CUDA_ERROR_CHECK(cudaMalloc(&d_sieve, num_words * sizeof(unsigned int)));

    int block_size = 256;
    int init_blocks = (num_words + block_size - 1) / block_size;

    sieve_init_v5<<<init_blocks, block_size>>>(d_sieve, num_words);

    // Mark 1 as not prime
    unsigned int first_word;
    CUDA_ERROR_CHECK(cudaMemcpy(&first_word, d_sieve, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    first_word &= ~1u;
    CUDA_ERROR_CHECK(cudaMemcpy(d_sieve, &first_word, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Get small primes on CPU
    int sqrt_limit = (int)sqrt((double)limit) + 1;
    int small_size = (sqrt_limit + 1) / 2;
    unsigned char *small_sieve = (unsigned char *)malloc(small_size);
    memset(small_sieve, 1, small_size);
    small_sieve[0] = 0;

    for (int i = 1; i < small_size; i++) {
        int p = 2 * i + 1;
        if (p * p > sqrt_limit) break;
        if (small_sieve[i]) {
            for (int j = (p * p - 1) / 2; j < small_size; j += p) {
                small_sieve[j] = 0;
            }
        }
    }

    // Collect small primes
    int *small_primes = (int *)malloc(small_size * sizeof(int));
    int num_small = 0;
    for (int i = 1; i < small_size; i++) {
        if (small_sieve[i]) {
            int p = 2 * i + 1;
            if ((long long)p * p <= limit) {
                small_primes[num_small++] = p;
            }
        }
    }

    // Copy small primes to device
    int *d_primes;
    CUDA_ERROR_CHECK(cudaMalloc(&d_primes, num_small * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_primes, small_primes, num_small * sizeof(int), cudaMemcpyHostToDevice));

    // Launch 2D grid: x dimension for marking, y dimension for different primes
    int primes_per_launch = 64;
    for (int i = 0; i < num_small; i += primes_per_launch) {
        int batch = min(primes_per_launch, num_small - i);
        dim3 grid(256, batch);
        sieve_mark_multi<<<grid, block_size>>>(d_sieve, sieve_bits, d_primes, num_small, i);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Collect primes
    unsigned int *h_sieve = (unsigned int *)malloc(num_words * sizeof(unsigned int));
    CUDA_ERROR_CHECK(cudaMemcpy(h_sieve, d_sieve, num_words * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int prime_count = 0;
    h_primes[prime_count++] = 2;

    for (int word = 0; word < num_words && prime_count < max_primes; word++) {
        unsigned int w = h_sieve[word];
        while (w && prime_count < max_primes) {
            int bit = __builtin_ctz(w);
            int idx = word * BITS_PER_WORD + bit;
            int num = 2 * idx + 1;
            if (num <= limit) {
                h_primes[prime_count++] = num;
            }
            w &= w - 1;
        }
    }

    free(small_sieve);
    free(small_primes);
    free(h_sieve);
    CUDA_ERROR_CHECK(cudaFree(d_sieve));
    CUDA_ERROR_CHECK(cudaFree(d_primes));
}

//=============================================================================
// VERSION 7: ULTIMATE - Wheel factorization (2,3,5) + coalesced writes
//=============================================================================
// Wheel pattern for mod 30: only 8 out of 30 numbers can be prime (coprime to 2,3,5)
// Numbers: 1,7,11,13,17,19,23,29 (mod 30)
__constant__ int d_wheel_offsets[8] = {1, 7, 11, 13, 17, 19, 23, 29};
const int h_wheel_offsets[8] = {1, 7, 11, 13, 17, 19, 23, 29};  // Host copy

// Mark composites for a prime using wheel factorization
__global__ void sieve_mark_wheel(unsigned char *sieve, long long sieve_size, int prime) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long start = (long long)prime * prime;

    // Find first multiple >= start that's on the wheel
    long long num = start + tid * prime;

    // We store only numbers coprime to 30
    // For a number n coprime to 30: index = (n/30)*8 + wheel_position
    while (num < sieve_size * 30 / 8 + 30) {
        int mod30 = num % 30;
        // Check if this number is on the wheel (coprime to 30)
        int wheel_pos = -1;
        if (mod30 == 1) wheel_pos = 0;
        else if (mod30 == 7) wheel_pos = 1;
        else if (mod30 == 11) wheel_pos = 2;
        else if (mod30 == 13) wheel_pos = 3;
        else if (mod30 == 17) wheel_pos = 4;
        else if (mod30 == 19) wheel_pos = 5;
        else if (mod30 == 23) wheel_pos = 6;
        else if (mod30 == 29) wheel_pos = 7;

        if (wheel_pos >= 0) {
            long long idx = (num / 30) * 8 + wheel_pos;
            if (idx > 0 && idx < sieve_size) {  // Skip index 0 (represents 1)
                sieve[idx] = 0;
            }
        }
        num += prime * gridDim.x * blockDim.x;
    }
}

void find_primes_v7(int max_primes, int *h_primes, bool verbose) {
    int limit;
    if (max_primes < 6) {
        limit = 15;
    } else {
        double n = (double)max_primes;
        limit = (int)(n * (log(n) + log(log(n)))) + 100;
    }

    // Wheel sieve: only store numbers coprime to 30 (2,3,5)
    // 8 numbers per 30 integers = 26.7% of original size
    long long sieve_size = ((long long)limit / 30 + 1) * 8;

    unsigned char *d_sieve;
    CUDA_ERROR_CHECK(cudaMalloc(&d_sieve, sieve_size));
    CUDA_ERROR_CHECK(cudaMemset(d_sieve, 1, sieve_size));

    // Mark index 0 (represents 1) as not prime
    unsigned char zero = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(d_sieve, &zero, 1, cudaMemcpyHostToDevice));

    int block_size = 256;
    int sqrt_limit = (int)sqrt((double)limit) + 1;

    // CPU sieve for small primes
    int small_size = sqrt_limit + 1;
    bool *small_sieve = (bool *)calloc(small_size, sizeof(bool));
    for (int i = 2; i < small_size; i++) small_sieve[i] = true;

    for (int p = 2; p * p < small_size; p++) {
        if (small_sieve[p]) {
            for (int m = p * p; m < small_size; m += p) {
                small_sieve[m] = false;
            }
        }
    }

    // Mark composites on GPU for primes > 5
    for (int p = 7; p <= sqrt_limit; p++) {
        if (small_sieve[p]) {
            long long start = (long long)p * p;
            long long count = (limit - start) / p + 1;
            int mark_blocks = min((long long)1024, (count + block_size - 1) / block_size);
            sieve_mark_wheel<<<mark_blocks, block_size>>>(d_sieve, sieve_size, p);
        }
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy back
    unsigned char *h_sieve = (unsigned char *)malloc(sieve_size);
    CUDA_ERROR_CHECK(cudaMemcpy(h_sieve, d_sieve, sieve_size, cudaMemcpyDeviceToHost));

    // Collect primes
    int prime_count = 0;
    h_primes[prime_count++] = 2;
    h_primes[prime_count++] = 3;
    h_primes[prime_count++] = 5;

    for (long long i = 1; i < sieve_size && prime_count < max_primes; i++) {
        if (h_sieve[i]) {
            int wheel_pos = i % 8;
            long long num = (i / 8) * 30 + h_wheel_offsets[wheel_pos];
            if (num <= limit && num > 5) {
                h_primes[prime_count++] = (int)num;
            }
        }
    }

    free(small_sieve);
    free(h_sieve);
    CUDA_ERROR_CHECK(cudaFree(d_sieve));
}

//=============================================================================
// VERSION 8: V6 + STREAMS - Async multi-stream execution
//=============================================================================
#define NUM_STREAMS 4

void find_primes_v8(int max_primes, int *h_primes, bool verbose) {
    int limit;
    if (max_primes < 6) {
        limit = 15;
    } else {
        double n = (double)max_primes;
        limit = (int)(n * (log(n) + log(log(n)))) + 100;
    }

    int sieve_bits = (limit + 1) / 2;
    int num_words = (sieve_bits + BITS_PER_WORD - 1) / BITS_PER_WORD;

    unsigned int *d_sieve;
    CUDA_ERROR_CHECK(cudaMalloc(&d_sieve, num_words * sizeof(unsigned int)));

    int block_size = 256;
    int init_blocks = (num_words + block_size - 1) / block_size;

    sieve_init_v5<<<init_blocks, block_size>>>(d_sieve, num_words);

    // Mark 1 as not prime
    unsigned int first_word;
    CUDA_ERROR_CHECK(cudaMemcpy(&first_word, d_sieve, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    first_word &= ~1u;
    CUDA_ERROR_CHECK(cudaMemcpy(d_sieve, &first_word, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Get small primes on CPU
    int sqrt_limit = (int)sqrt((double)limit) + 1;
    int small_size = (sqrt_limit + 1) / 2;
    unsigned char *small_sieve = (unsigned char *)malloc(small_size);
    memset(small_sieve, 1, small_size);
    small_sieve[0] = 0;

    for (int i = 1; i < small_size; i++) {
        int p = 2 * i + 1;
        if (p * p > sqrt_limit) break;
        if (small_sieve[i]) {
            for (int j = (p * p - 1) / 2; j < small_size; j += p) {
                small_sieve[j] = 0;
            }
        }
    }

    // Collect small primes
    int *small_primes = (int *)malloc(small_size * sizeof(int));
    int num_small = 0;
    for (int i = 1; i < small_size; i++) {
        if (small_sieve[i]) {
            int p = 2 * i + 1;
            if ((long long)p * p <= limit) {
                small_primes[num_small++] = p;
            }
        }
    }

    // Copy small primes to device
    int *d_primes;
    CUDA_ERROR_CHECK(cudaMalloc(&d_primes, num_small * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_primes, small_primes, num_small * sizeof(int), cudaMemcpyHostToDevice));

    // Create streams for async execution
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Distribute primes across streams
    int primes_per_launch = 32;
    int stream_idx = 0;
    for (int i = 0; i < num_small; i += primes_per_launch) {
        int batch = min(primes_per_launch, num_small - i);
        dim3 grid(256, batch);
        sieve_mark_multi<<<grid, block_size, 0, streams[stream_idx]>>>(
            d_sieve, sieve_bits, d_primes, num_small, i);
        stream_idx = (stream_idx + 1) % NUM_STREAMS;
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_ERROR_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }

    // Collect primes
    unsigned int *h_sieve = (unsigned int *)malloc(num_words * sizeof(unsigned int));
    CUDA_ERROR_CHECK(cudaMemcpy(h_sieve, d_sieve, num_words * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int prime_count = 0;
    h_primes[prime_count++] = 2;

    for (int word = 0; word < num_words && prime_count < max_primes; word++) {
        unsigned int w = h_sieve[word];
        while (w && prime_count < max_primes) {
            int bit = __builtin_ctz(w);
            int idx = word * BITS_PER_WORD + bit;
            int num = 2 * idx + 1;
            if (num <= limit) {
                h_primes[prime_count++] = num;
            }
            w &= w - 1;
        }
    }

    free(small_sieve);
    free(small_primes);
    free(h_sieve);
    CUDA_ERROR_CHECK(cudaFree(d_sieve));
    CUDA_ERROR_CHECK(cudaFree(d_primes));
}

//=============================================================================
// MAIN
//=============================================================================
int main(int argc, char **argv) {
    int max_primes = 100000;
    int version = 0;

    if (argc > 1) max_primes = atoi(argv[1]);
    if (argc > 2) version = atoi(argv[2]);

    printf("=== CUDA Prime Finder - Performance Comparison ===\n");
    printf("Finding first %d primes\n\n", max_primes);

    int *h_primes = (int *)malloc(max_primes * sizeof(int));

    auto run_version = [&](int v, const char *name, void (*func)(int, int *, bool)) {
        memset(h_primes, 0, max_primes * sizeof(int));

        // Warmup
        func(min(1000, max_primes), h_primes, false);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        // Timed run with CUDA events for accurate GPU timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        func(max_primes, h_primes, false);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        printf("Version %d %-20s: %8.3f ms\n", v, name, ms);
        printf("  First 10: ");
        for (int i = 0; i < min(10, max_primes); i++) printf("%d ", h_primes[i]);
        printf("\n  Last prime (#%d): %d\n\n", max_primes, h_primes[max_primes - 1]);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    if (version == 0 || version == 1) {
        if (max_primes <= 20000) {
            run_version(1, "(Baseline)", find_primes_v1);
        } else {
            printf("Version 1 (Baseline): SKIPPED (too slow for %d primes)\n\n", max_primes);
        }
    }

    if (version == 0 || version == 2) {
        run_version(2, "(Batch)", find_primes_v2);
    }

    if (version == 0 || version == 3) {
        run_version(3, "(Basic Sieve)", find_primes_v3);
    }

    if (version == 0 || version == 4) {
        run_version(4, "(Odd-only Sieve)", find_primes_v4);
    }

    if (version == 0 || version == 5) {
        run_version(5, "(Bit-packed)", find_primes_v5);
    }

    if (version == 0 || version == 6) {
        run_version(6, "(Multi-prime)", find_primes_v6);
    }

    if (version == 0 || version == 7) {
        run_version(7, "(Wheel 2,3,5)", find_primes_v7);
    }

    if (version == 0 || version == 8) {
        run_version(8, "(V6+Streams)", find_primes_v8);
    }

    // Summary
    printf("=== Performance Summary ===\n");
    printf("V1: Baseline (1 kernel/candidate) - High overhead, O(n) launches\n");
    printf("V2: Batch processing - Fewer launches, but still O(n/batch)\n");
    printf("V3: Basic sieve - O(sqrt(limit)) launches\n");
    printf("V4: Odd-only sieve - 50%% memory reduction\n");
    printf("V5: Bit-packed - 8x memory reduction, better cache\n");
    printf("V6: Multi-prime marking - Better GPU utilization\n");
    printf("V7: Wheel (2,3,5) - 73%% memory reduction\n");
    printf("V8: V6 + CUDA streams for async\n");

    free(h_primes);
    return 0;
}
