#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>

// gcc -o build/mmap_msgstore mmap_msgstore.c -Wall && ./build/mmap_msgstore

// TLV (Type-Length-Value) pattern for variable-length messages:
// each message is stored as: [fixed header] [variable body]
// header contains the body length, so reader knows how many bytes to read next
//
// file layout:
// ┌──header──┐┌──body──┐┌──header──┐┌──body──┐┌──header──┐┌──body──┐
// │id│len│ts ││ bytes  ││id│len│ts ││ bytes  ││ (zeros)  ││        │
// └──20B─────┘└─varlen─┘└──20B─────┘└─varlen─┘└──empty───┘└────────┘
//
// struct.pack is only needed for numbers (int → bytes)
// raw byte data (strings, blobs) can be written directly — any length

#define MAX_FILESIZE (1024 * 1024) // 1MB for demo
#define DB_FILE "/tmp/mmap_msgstore.db"

typedef struct {
    unsigned long msg_id;    // 8 bytes
    unsigned int length;     // 4 bytes — length of body that follows
    unsigned long timestamp; // 8 bytes — milliseconds
} __attribute__((packed)) MsgHeader; // 20 bytes (packed to avoid padding)

typedef struct {
    char *base;
    size_t offset;
    size_t capacity;
} MsgStore;

MsgStore *store_open(const char *path, int truncate) {
    int flags = O_RDWR | O_CREAT | (truncate ? O_TRUNC : 0);
    int fd = open(path, flags, 0644);
    if (fd < 0) { perror("open"); return NULL; }
    ftruncate(fd, MAX_FILESIZE);

    char *base = mmap(NULL, MAX_FILESIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { perror("mmap"); return NULL; }

    MsgStore *s = malloc(sizeof(MsgStore));
    s->base = base;
    s->offset = 0;
    s->capacity = MAX_FILESIZE;
    return s;
}

void store_close(MsgStore *s) {
    msync(s->base, s->capacity, MS_SYNC);
    munmap(s->base, s->capacity);
    free(s);
}

int store_write(MsgStore *s, unsigned long msg_id, const char *body, unsigned int body_len) {
    if (s->offset + sizeof(MsgHeader) + body_len > s->capacity) {
        printf("no space left\n");
        return -1;
    }

    // write fixed header at current offset
    MsgHeader *hdr = (MsgHeader *)(s->base + s->offset);
    hdr->msg_id = msg_id;
    hdr->length = body_len;
    hdr->timestamp = (unsigned long)(time(NULL)) * 1000;

    // write variable-length body right after header
    memcpy(s->base + s->offset + sizeof(MsgHeader), body, body_len);

    s->offset += sizeof(MsgHeader) + body_len;
    return 0;
}

// returns pointer to body (zero-copy read), fills header info
const char *store_read(MsgStore *s, MsgHeader *hdr_out) {
    if (s->offset + sizeof(MsgHeader) > s->capacity) return NULL;

    MsgHeader *hdr = (MsgHeader *)(s->base + s->offset);
    if (hdr->length == 0) return NULL; // no more messages

    *hdr_out = *hdr;
    const char *body = s->base + s->offset + sizeof(MsgHeader);
    s->offset += sizeof(MsgHeader) + hdr->length;
    return body;
}

int main() {
    printf("MsgHeader size: %zu bytes\n\n", sizeof(MsgHeader));

    // write messages of different sizes
    MsgStore *s = store_open(DB_FILE, 1); // truncate = 1, fresh file

    store_write(s, 1, "hello", 5);
    store_write(s, 2, "this is a longer message with more content", 43);
    store_write(s, 3, "short", 5);

    // write a large message
    char big[500];
    memset(big, 'X', 500);
    store_write(s, 4, big, 500);

    printf("=== wrote 4 messages, total %zu bytes used ===\n\n", s->offset);
    store_close(s);

    // re-open and read all messages back (truncate = 0, keep data)
    printf("=== re-open and read ===\n");
    s = store_open(DB_FILE, 0);

    MsgHeader hdr;
    const char *body;
    while ((body = store_read(s, &hdr)) != NULL) {
        printf("msg_id=%-3lu len=%-4u ts=%lu body=\"%.40s%s\"\n",
               hdr.msg_id, hdr.length, hdr.timestamp,
               body, hdr.length > 40 ? "..." : "");
    }

    store_close(s);
    unlink(DB_FILE);
    return 0;
}
