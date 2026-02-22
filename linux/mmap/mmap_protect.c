// mprotect + SIGSEGV: catch illegal memory writes at the page level
// this is how copy-on-write, dirty page tracking, and garbage collectors work
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>

// gcc -o build/mmap_protect mmap_protect.c -Wall && ./build/mmap_protect

// mprotect(addr, len, prot) — change page permissions at runtime
//   PROT_READ, PROT_WRITE, PROT_EXEC, PROT_NONE
//   addr must be page-aligned, applies to whole pages
//
// writing to a read-only page triggers SIGSEGV
// with a signal handler (SA_SIGINFO), you can:
//   1. get the fault address from siginfo_t->si_addr
//   2. unprotect that page with mprotect()
//   3. execution resumes at the faulting instruction (now succeeds)
//
// real-world uses of this pattern:
//   copy-on-write: fork shares pages read-only, write fault → copy → private page
//   dirty page tracking: databases detect which pages need flushing
//   GC write barriers: garbage collectors detect pointer mutations
//   software transactional memory: detect write conflicts

// Copy-on-write (COW) means: share until someone writes, then copy. (eg, fork)

static long page_size;
static void *protected_region;
static size_t protected_size;
static int fault_count = 0;

// signal handler: called when we write to a read-only page
void sigsegv_handler(int sig, siginfo_t *info, void *ctx) {
    void *fault_addr = info->si_addr;
    // find which page was accessed
    long page_offset = ((char *)fault_addr - (char *)protected_region) / page_size;

    printf("  [SEGV] write fault at %p (page %ld) — unprotecting\n",
           fault_addr, page_offset);
    fault_count++;

    // fix the fault: make that page writable so execution can continue
    void *page_start = (void *)((long)fault_addr & ~(page_size - 1));
    mprotect(page_start, page_size, PROT_READ | PROT_WRITE);
    // execution resumes at the faulting instruction, which now succeeds
}

int main() {
    page_size = sysconf(_SC_PAGESIZE);
    protected_size = page_size * 4;

    printf("page size: %ld bytes, protecting %zu bytes (4 pages)\n\n", page_size, protected_size);

    // allocate anonymous memory
    protected_region = mmap(NULL, protected_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (protected_region == MAP_FAILED) { perror("mmap"); return 1; }

    // write initial data
    memset(protected_region, 0, protected_size);

    // install SIGSEGV handler with SA_SIGINFO to get fault address
    struct sigaction sa = {
        .sa_sigaction = sigsegv_handler,
        .sa_flags = SA_SIGINFO,
    };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);

    // now mark all pages as READ-ONLY
    mprotect(protected_region, protected_size, PROT_READ);
    printf("=== all 4 pages set to read-only ===\n");
    printf("reading works fine: byte[0] = %d\n\n", ((char *)protected_region)[0]);

    // writing to different pages triggers SIGSEGV, our handler catches it
    printf("=== writing to page 0 ===\n");
    ((char *)protected_region)[0] = 'A';
    printf("  write succeeded (handler unprotected the page)\n\n");

    printf("=== writing to page 2 ===\n");
    ((char *)protected_region)[2 * page_size] = 'B';
    printf("  write succeeded\n\n");

    printf("=== writing to page 0 again ===\n");
    ((char *)protected_region)[1] = 'C';
    printf("  write succeeded (no fault — page already unprotected)\n\n");

    printf("=== writing to page 1 and 3 ===\n");
    ((char *)protected_region)[1 * page_size] = 'D';
    ((char *)protected_region)[3 * page_size] = 'E';
    printf("  both writes succeeded\n\n");

    printf("total faults caught: %d (pages 0, 2, 1, 3 = 4 faults)\n", fault_count);
    printf("page 0 was written twice but only faulted once — \n");
    printf("this is how the kernel tracks dirty pages!\n");

    munmap(protected_region, protected_size);
    return 0;
}
