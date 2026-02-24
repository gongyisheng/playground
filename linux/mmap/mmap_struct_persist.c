// mmap as a mini database: persist structs directly to a file
// no serialization needed — the struct layout IS the file format
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// gcc -o build/mmap_struct_persist mmap_struct_persist.c -Wall && ./build/mmap_struct_persist

// MAP_SHARED makes the mmap'd region a direct window into the file
// writing to the pointer writes to the file — no write() syscall needed
// the struct's in-memory layout IS the file format — zero serialization
// tradeoff: not portable across architectures (endianness, padding differ)
// this pattern is used by SQLite WAL, Redis RDB, and many embedded databases
//
// ftruncate(fd, size): ensure file is large enough before mmap
// msync(addr, len, MS_SYNC): flush changes to disk (otherwise kernel flushes lazily)

#define MAX_USERS 100
#define DB_FILE "/tmp/mmap_users.db"

typedef struct {
    int id;
    char name[32];
    float score;
} User;

typedef struct {
    int count;
    User users[MAX_USERS];
} UserDB;

// map the db file into memory, creating it if needed
UserDB *db_open(const char *path) {
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) { perror("open"); return NULL; }

    // ensure file is large enough to hold the struct
    ftruncate(fd, sizeof(UserDB));

    // MAP_SHARED: writes to the pointer go directly to the file
    UserDB *db = mmap(NULL, sizeof(UserDB), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (db == MAP_FAILED) { perror("mmap"); return NULL; }
    return db;
}

void db_close(UserDB *db) {
    msync(db, sizeof(UserDB), MS_SYNC); // flush to disk
    munmap(db, sizeof(UserDB));
}

void db_add(UserDB *db, int id, const char *name, float score) {
    if (db->count >= MAX_USERS) { printf("db full\n"); return; }
    User *u = &db->users[db->count++];
    u->id = id;
    strncpy(u->name, name, 31);
    u->score = score;
    // no write() call needed — the data is already on disk via mmap
}

void db_print(UserDB *db) {
    printf("UserDB: %d users (struct size: %zu bytes)\n", db->count, sizeof(UserDB));
    for (int i = 0; i < db->count; i++) {
        User *u = &db->users[i];
        printf("  [%d] id=%d name=%-10s score=%.1f\n", i, u->id, u->name, u->score);
    }
}

int main(int argc, char *argv[]) {
    // first run: create and populate
    printf("=== writing to %s ===\n", DB_FILE);
    UserDB *db = db_open(DB_FILE);
    db->count = 0; // reset
    db_add(db, 1, "Alice", 95.5);
    db_add(db, 2, "Bob", 87.3);
    db_add(db, 3, "Charlie", 91.0);
    db_print(db);
    db_close(db);

    // second open: data persists without any parsing
    printf("\n=== re-opening (data persisted via mmap) ===\n");
    db = db_open(DB_FILE);
    db_print(db);

    // modify in place
    db->users[1].score = 99.9;
    printf("\n=== modified Bob's score in-place ===\n");
    db_print(db);
    db_close(db);

    // third open: modification persisted
    printf("\n=== re-opening (modification persisted) ===\n");
    db = db_open(DB_FILE);
    db_print(db);
    db_close(db);

    printf("\nfile size on disk: ");
    fflush(stdout);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "ls -l %s | awk '{print $5, \"bytes\"}'", DB_FILE);
    system(cmd);

    return 0;
}
