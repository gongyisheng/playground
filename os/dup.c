#include <stdio.h>
#include <unistd.h>

// This is an example of how to duplicate a file descriptor
// Two file descriptors share an offset if they were derived 
// from the same original file descriptor by a sequence of fork 
// and dup calls
int main(){
    int fd = dup(1);
    write(1, "hello ", 6);
    write(fd, "world\n", 6);
    return 0;
}