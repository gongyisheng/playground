#include <stdio.h>
#include <unistd.h> // for fork(), write()
#include <fcntl.h> // for open()
#include <stdlib.h> // for waitpid(), exit()

int main() {
    pid_t pid = fork();
    if(pid == 0) {
        // Child process
        write(1, "hello ", 6); // Print "hello" to the console
        exit(0); // Exit the child process
    } else {
        // Parent process
        int status;
        waitpid(pid, &status, 0); // Wait for the child process to finish
        write(1, "world\n", 6); // Print "world" to the console
    }
    return 0;
}