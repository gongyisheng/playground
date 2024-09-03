#include <stdio.h>
#include <malloc.h>

int main(int argc, char *argv[])
{
    int i;
    void *data[10];

    printf("Hello World\n");

    for (i = 0; i < 10; ++i) {
        data[i] = malloc(i+10);
    }

    for (i = 0; i < 9; ++i) {
        free(data[i]);
    }

    return 0;
}