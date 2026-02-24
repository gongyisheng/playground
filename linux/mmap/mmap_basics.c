#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>

// gcc -o build/mmap_basics mmap_basics.c -Wall && ./build/mmap_basics

// mmap() — <sys/mman.h>
// maps a file (or anonymous memory) into process virtual address space
// instead of read()/write() syscalls, access file content through pointers
//
// void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
//   addr:   hint address, usually NULL (let kernel choose)
//   length: bytes to map
//   prot:   PROT_READ, PROT_WRITE, PROT_EXEC
//   flags:  MAP_SHARED (changes go to file) vs MAP_PRIVATE (copy-on-write)
//   fd:     file descriptor, -1 for anonymous
//   offset: offset into file, must be page-aligned
//   returns: pointer to mapped region, or MAP_FAILED (not NULL!) on error
//
// int munmap(void *addr, size_t length);  — unmap when done
// int msync(void *addr, size_t length, int flags);  — flush to disk
//   MS_SYNC: block until done, MS_ASYNC: schedule and return
//
// traditional I/O: disk → page cache → user buffer (2 copies)
// mmap: disk → page cache, process reads page cache directly (zero-copy)
//
// open() flags — <fcntl.h> (bit flags, combine with |)
//   O_RDONLY, O_WRONLY, O_RDWR: access mode
//   O_CREAT: create if not exists (needs permission arg, e.g. 0644)
//   O_TRUNC: truncate to 0 if exists
//   O_EXCL:  fail if already exists (good for lock files)
//
// common use cases:
//   1. file I/O: map file, read/write via pointers (databases, config parsers)
//   2. anonymous: MAP_ANONYMOUS allocates memory without file (how malloc works)
//   3. shared IPC: MAP_SHARED between processes for zero-copy communication

// 1: read a file using mmap (zero-copy)
void file_read(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return; }

    struct stat st;
    fstat(fd, &st);

    // map entire file into memory as read-only
    char *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // fd can be closed after mmap
    if (mapped == MAP_FAILED) { perror("mmap"); return; }

    printf("=== file read via mmap ===\n");
    printf("file size: %ld bytes\n", st.st_size);
    printf("first 80 chars:\n%.80s\n\n", mapped);

    munmap(mapped, st.st_size);
}

// demo 2: modify a file in-place using mmap
void file_write(const char *path) {
    // create a temp file with some content
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return; }

    const char *initial = "Hello, mmap world! This text lives on disk.";
    size_t len = strlen(initial);
    write(fd, initial, len);

    // map file as read-write, MAP_SHARED so changes persist to disk
    char *mapped = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { perror("mmap"); return; }

    printf("=== file write via mmap ===\n");
    printf("before: %.*s\n", (int)len, mapped);

    // memcpy can write to any offset in the mmap'd region
    memcpy(mapped, "HELLO", 5);              // start: "HELLO, mmap world!..."
    memcpy(mapped + 7, "LINUX", 5);          // middle: "HELLO, LINUX world!..."
    memcpy(mapped + 19, "on memory.", 10);   // offset: "HELLO, LINUX world on memory...."
    // msync ensures changes are flushed to disk
    msync(mapped, len, MS_SYNC);

    printf("after:  %.*s\n\n", (int)len, mapped);
    munmap(mapped, len);
}

// demo 3: anonymous mmap (memory allocation without a file)
void anonymous() {
    size_t size = 4096; // one page

    // MAP_ANONYMOUS + fd=-1: no file backing, just raw memory
    // this is what malloc() uses internally for large allocations
    int *arr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arr == MAP_FAILED) { perror("mmap"); return; }

    printf("=== anonymous mmap ===\n");
    int count = size / sizeof(int);
    for (int i = 0; i < count; i++) arr[i] = i * i;
    printf("arr[0]=%d, arr[10]=%d, arr[100]=%d\n", arr[0], arr[10], arr[100]);
    printf("total ints in one page: %d\n\n", count);

    munmap(arr, size);
}

// demo 4: shared memory between parent and child process
void shared_memory() {
    size_t size = 4096;

    // MAP_SHARED + MAP_ANONYMOUS: shared between fork()'d processes
    int *shared = mmap(NULL, size, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) { perror("mmap"); return; }

    *shared = 0;
    printf("=== shared memory IPC ===\n");
    fflush(stdout); // flush before fork to avoid duplicate output

    pid_t pid = fork();
    if (pid == 0) {
        // child: write to shared memory
        *shared = 42;
        printf("child wrote: %d\n", *shared);
        _exit(0); // _exit avoids flushing parent's buffered stdout
    }

    // parent: wait and read
    waitpid(pid, NULL, 0);
    printf("parent read:  %d (written by child via shared mmap)\n\n", *shared);

    munmap(shared, size);
}

int main(int argc, char *argv[]) {
    // read this source file itself
    file_read(argv[0][0] == '/' ? __FILE__ : "mmap_demo.c");
    file_write("/tmp/mmap_test.txt");
    anonymous();
    shared_memory();

    // verify the file write persisted to disk
    printf("=== verify persistence ===\n");
    int fd = open("/tmp/mmap_test.txt", O_RDONLY);
    if (fd >= 0) {
        char buf[256];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        if (n > 0) { buf[n] = '\0'; printf("/tmp/mmap_test.txt: %s\n", buf); }
        close(fd);
    }

    return 0;
}
