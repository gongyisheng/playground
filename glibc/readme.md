# Memory leak detection using glibc
compile program: `gcc memory_leak.c -o bin/memory_leak`  
run profiling:
```
yisheng@rpi5:/media/hdddisk/playground/glibc$ memusage bin/memory_leak
Hello World

Memory usage summary: heap total: 1169, heap peak: 1169, stack peak: 432
         total calls   total memory   failed calls
 malloc|         11           1169              0
realloc|          0              0              0  (nomove:0, dec:0, free:0)
 calloc|          0              0              0
   free|          9            126
Histogram for block sizes:
    0-15              6  54% ==================================================
   16-31              4  36% =================================
 1024-1039            1   9% ========
```
