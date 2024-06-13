#include <stdio.h>
#include <fcntl.h> // for open()
#include <unistd.h> // for close()
#include <errno.h> // for errno
#include <string.h> // for strerror()

int main() {
    // Path to the file you want to open
    const char *filePath = "example.txt";

    // Open the file in read-only mode
    int fileDescriptor = open(filePath, O_RDONLY);

    // Check if open was successful
    if (fileDescriptor == -1) {
        // If not, print the error message
        fprintf(stderr, "Error opening the file %s: %s\n", filePath, strerror(errno));
        return 1;
    }

    printf("Successfully opened %s with file descriptor %d\n", filePath, fileDescriptor);

    // Close the file after use
    if (close(fileDescriptor) == -1) {
        // If close fails, print the error message
        fprintf(stderr, "Error closing the file %s: %s\n", filePath, strerror(errno));
        return 1;
    }

    printf("Successfully closed %s\n", filePath);

    return 0;
}