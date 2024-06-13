#include <stdio.h>
#include <unistd.h>

int main(){
    char *argv[4];
    argv[0] = ""; // This is the name of the program
    argv[1] = "echo"; // This is the first argument
    argv[2] = "hello"; // This is the second argument
    argv[3] = 0; // This is the end of the arguments
    execv("/bin/echo", argv);
    printf("exec error\n");
    return 1;
}