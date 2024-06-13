#include <stdio.h>
#include <unistd.h>
#include <fcntl.h> // for open()
#include <unistd.h> // for close()

// This is an example of how to redirect the standard input of a program to a file
// This program will execute the cat command and read the content of example.txt as the standard input
int main() {
    char *argv[2];
    argv[0] = "cat";
    argv[1] = 0;

    pid_t pid = fork();
    if(pid == 0) {
        close(0); // close stdin
        open("example.txt", O_RDONLY); // open file as stdin (fd = 0)
        execv("/bin/cat", argv); // execute cat, then you can see the content of example.txt as the console input
    }
    return 0;
}