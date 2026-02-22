// observe page faults with mincore(): see which pages are in RAM vs on disk
// demonstrates lazy loading — pages aren't loaded until you touch them
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// gcc -o build/mmap_pagefault mmap_pagefault.c -Wall && ./build/mmap_pagefault

// mmap is lazy — calling mmap() doesn't load the file into RAM
// kernel sets up page table entries pointing to "nothing"
// first access triggers a page fault → kernel loads that 4KB page from disk
// this is why mmap is fast for random access on huge files — only pay for pages you touch
//
// mincore(addr, length, vec) — query which pages are in physical RAM
//   vec[i] & 1 == 1: page is resident (in RAM)
//   vec[i] & 1 == 0: page is NOT resident (still on disk)
//
// madvise(addr, length, advice) — hint to kernel about access patterns
//   MADV_DONTNEED: evict pages (free memory, will re-fault from disk on next access)
//   MADV_SEQUENTIAL: will read sequentially (kernel prefetches ahead)
//   MADV_RANDOM: will read randomly (disable prefetch)
//
// posix_fadvise(fd, offset, len, advice) — same hints but on fd before mmap
//   POSIX_FADV_DONTNEED: drop pages from page cache

void show_resident(const char *label, void *addr, size_t length) {
    long page_size = sysconf(_SC_PAGESIZE);
    size_t pages = (length + page_size - 1) / page_size;
    unsigned char *vec = malloc(pages);

    // mincore: query which pages are currently in physical RAM
    // vec[i] & 1 == 1: page is resident (in RAM)
    // vec[i] & 1 == 0: page is NOT resident (still on disk)
    mincore(addr, length, vec);

    int resident = 0;
    for (size_t i = 0; i < pages; i++)
        if (vec[i] & 1) resident++;

    printf("%-25s %d/%zu pages in RAM", label, resident, pages);
    if (pages <= 32) {
        printf("  [");
        for (size_t i = 0; i < pages; i++)
            printf("%c", (vec[i] & 1) ? '#' : '.');
        printf("]");
    }
    printf("\n");
    free(vec);
}

int main() {
    long page_size = sysconf(_SC_PAGESIZE);
    printf("page size: %ld bytes\n\n", page_size);

    // use a sparse file: ftruncate sets the size but doesn't write any data
    // pages don't exist in page cache — they'll fault in as zero-filled on first read
    // (writing data with write()/pwrite() would populate the page cache,
    // making all pages resident before mmap even starts)
    const char *path = "/tmp/mmap_pagefault_test";
    size_t num_pages = 32; // small enough to show per-page bitmap
    size_t file_size = page_size * num_pages;
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    ftruncate(fd, file_size);

    char *mapped = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { perror("mmap"); return 1; }

    // disable readahead — otherwise kernel prefetches nearby pages on each fault
    madvise(mapped, file_size, MADV_RANDOM);

    // 1: right after mmap — nothing loaded yet
    show_resident("after mmap (no access)", mapped, file_size);

    // 2: touch page 0 — triggers a page fault, kernel allocates and zeros the page
    volatile char c = mapped[0];
    show_resident("after reading page 0", mapped, file_size);

    // 3: touch page 16 (middle)
    c = mapped[16 * page_size];
    show_resident("after reading page 16", mapped, file_size);

    // 4: touch pages 24-31
    for (int i = 24; i < 32; i++)
        c = mapped[i * page_size];
    show_resident("after reading pages 24-31", mapped, file_size);

    // 5: read everything
    for (size_t i = 0; i < file_size; i += page_size)
        c = mapped[i];
    show_resident("after reading all pages", mapped, file_size);

    // 6: tell kernel to evict with madvise
    madvise(mapped, file_size, MADV_DONTNEED);
    show_resident("after madvise(DONTNEED)", mapped, file_size);

    (void)c;
    munmap(mapped, file_size);
    unlink(path);
    return 0;
}
