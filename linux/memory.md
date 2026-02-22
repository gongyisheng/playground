# Linux Memory Management

## Virtual Memory
each process gets its own virtual address space (e.g. 0x0000 - 0xFFFF...)
virtual addresses are mapped to physical RAM via **page tables**
two processes can have the same virtual address pointing to different physical pages

## Pages
memory is managed in fixed-size **pages** (typically 4KB, check with `getconf PAGESIZE`)
page is the smallest unit the kernel allocates, maps, and protects

## Page Table
per-process mapping: virtual address → physical address + permissions (read/write/exec)
stored in hardware (MMU uses it for every memory access)
`fork()` copies the page table, not the actual memory

## Page Fault
CPU accesses a virtual address and the page table says "not available" or "no permission"
CPU interrupts the program → kernel resolves it → program resumes (never notices)

**minor fault**: page exists but isn't mapped yet → kernel updates page table (~microseconds)
**major fault**: page is on disk, not in RAM → kernel reads from disk (~milliseconds)
**protection fault**: permission violation (e.g. write to read-only) → SIGSEGV or kernel handles COW

page faults sound like errors but they are normal — the kernel uses them for lazy loading, COW, and demand paging

## Copy-on-Write (COW)
share until someone writes, then copy

1. `fork()` — kernel copies page table, marks all pages **read-only** in both processes
2. both read — same physical pages, zero cost
3. one writes — page fault → kernel copies just that page → writer gets private writable copy

a process with 1GB memory can `fork()` almost instantly (only page table ~KB is copied)
pages are duplicated only when actually modified

where COW shows up:
- `fork()` — the main use case
- `MAP_PRIVATE` mmap — reads share with page cache, writes get a private copy

## Dirty Pages
a page in RAM that has been **modified but not yet written back to disk**

clean page: RAM content == disk content (safe to discard anytime)
dirty page: RAM content != disk content (must write to disk before discarding)

the kernel tracks dirty pages to know:
- **what to flush** — periodic writeback (~5 seconds by default) syncs dirty pages to disk
- **what can be evicted** — when RAM is low, clean pages dropped immediately; dirty pages must be written first
- **what msync()/fsync() writes** — only dirty pages, not everything

this is why `madvise(MADV_DONTNEED)` won't evict dirty pages — kernel can't discard them without losing data

dirty page tracking works like `mmap_protect.c` demo internally:
mark pages read-only → catch write fault → mark page as dirty → allow the write

## Memory Layout of a Process
```
high address
┌──────────────┐
│    stack      │  local variables, function args (grows down)
│      ↓        │
│              │
│      ↑        │
│    heap       │  malloc/free (grows up)
├──────────────┤
│    bss        │  uninitialized globals (zeroed)
├──────────────┤
│    data       │  initialized globals
├──────────────┤
│    text       │  code (read-only + executable)
└──────────────┘
low address
```

## Where Things Live
```c
char *p = "hello";      // p (pointer) on stack, "hello" in .rodata (read-only, in binary)
char arr[] = "hello";   // arr (6 bytes copy) on stack, mutable
char *h = malloc(6);    // h (pointer) on stack, 6 bytes on heap
int x = 42;             // x on stack
static int y = 10;      // y in data section
```

## Key Syscalls
`mmap()` — map file or anonymous memory into virtual address space (see mmap/)
`munmap()` — unmap
`mprotect()` — change page permissions (read/write/exec) at runtime
`madvise()` — hint to kernel about access patterns (MADV_RANDOM, MADV_DONTNEED, etc.)
`mincore()` — query which pages are currently in physical RAM
`brk()/sbrk()` — expand heap (used by malloc internally, rarely called directly)
