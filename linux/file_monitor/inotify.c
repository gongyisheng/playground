#include <stdio.h>
#include <stdlib.h>
#include <sys/inotify.h>

int main() {
    int fd, wd;
    fd = inotify_init();
    if (fd < 0) {
        perror("inotify_init");
        exit(EXIT_FAILURE);
    }

    wd = inotify_add_watch(fd, "/path/to/directory", IN_MODIFY | IN_CREATE | IN_DELETE);
    if (wd < 0) {
        perror("inotify_add_watch");
        exit(EXIT_FAILURE);
    }

    printf("Watching directory for file changes...\n");

    while (1) {
        char buf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
        struct inotify_event *event;
        int len = read(fd, buf, sizeof(buf));

        if (len < 0) {
            perror("read");
            exit(EXIT_FAILURE);
        }

        for (char *ptr = buf; ptr < buf + len; ptr += sizeof(struct inotify_event) + event->len) {
            event = (struct inotify_event *)ptr;

            if (event->mask & IN_CREATE) {
                printf("File created: %s\n", event->name);
            }
            if (event->mask & IN_MODIFY) {
                printf("File modified: %s\n", event->name);
            }
            if (event->mask & IN_DELETE) {
                printf("File deleted: %s\n", event->name);
            }
        }
    }

    close(wd);
    close(fd);

    return 0;
}