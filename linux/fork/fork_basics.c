#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

// gcc -o build/fork_basics fork_basics.c -Wall && ./build/fork_basics

// fork() — <unistd.h>
// creates a child process by duplicating the parent
// returns: child pid to parent, 0 to child, -1 on failure
//
// memory after fork: copy-on-write (COW)
// - kernel copies page table (cheap), marks all pages read-only
// - both read: same physical pages, no cost
// - one writes: page fault → kernel copies just that page → writer gets private copy
// - same virtual address can hold different values in parent vs child
//
// _exit() vs exit():
// - _exit(): terminate immediately, no stdio flush (safe in child)
// - exit(): flushes stdio buffers — can cause double output after fork
//
// stdout buffering:
// - terminal: line-buffered (flushes on \n)
// - piped/redirected: fully buffered (flushes on buffer full or exit)
// - setvbuf(stdout, NULL, _IOLBF, 0) forces line buffering

// 1: basic fork — parent and child diverge
void basic_fork() {
    printf("=== basic fork ===\n");
    printf("before fork: pid=%d\n", getpid());
    fflush(stdout);

    pid_t pid = fork();
    // from here, TWO processes are running the same code

    if (pid < 0) {
        perror("fork");
    } else if (pid == 0) {
        // child: fork() returns 0
        printf("  child:  pid=%d, parent=%d\n", getpid(), getppid());
        _exit(0);
    } else {
        // parent: fork() returns child's pid
        printf("  parent: pid=%d, child=%d\n", getpid(), pid);
        waitpid(pid, NULL, 0);
    }
    printf("\n");
}

// 2: copy-on-write — child gets a COPY, not a reference
void cow_demo() {
    printf("=== copy-on-write ===\n");
    int x = 100;
    printf("before fork: x=%d (addr=%p)\n", x, (void *)&x);
    fflush(stdout);

    pid_t pid = fork();
    if (pid == 0) {
        x = 999;
        // same virtual address, different physical page after write
        printf("  child modified:  x=%d (addr=%p)\n", x, (void *)&x);
        _exit(0);
    }
    waitpid(pid, NULL, 0);
    // parent's x is untouched — child wrote to its own copy
    printf("  parent original: x=%d (addr=%p)\n", x, (void *)&x);
    printf("  same virtual addr, different physical page!\n\n");
}

// 3: exit status — parent checks how child exited
void exit_status_demo() {
    printf("=== exit status ===\n");
    fflush(stdout);

    pid_t pid = fork();
    if (pid == 0) {
        printf("  child exiting with code 42\n");
        _exit(42);
    }

    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        printf("  parent: child exited with code %d\n", WEXITSTATUS(status));
    }
    printf("\n");
}

// 4: multiple children
void multi_child() {
    printf("=== multiple children ===\n");
    fflush(stdout);

    for (int i = 0; i < 3; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            printf("  child %d: pid=%d\n", i, getpid());
            _exit(i);
        }
    }

    // parent waits for all children
    int status;
    pid_t pid;
    while ((pid = wait(&status)) > 0) {
        printf("  child pid=%d exited with code %d\n", pid, WEXITSTATUS(status));
    }
    printf("\n");
}

int main() {
    // when piped, stdout is fully buffered — child's _exit() won't flush
    // line buffering ensures each printf with \n is visible immediately
    setvbuf(stdout, NULL, _IOLBF, 0);

    basic_fork();
    cow_demo();
    exit_status_demo();
    multi_child();
    return 0;
}
