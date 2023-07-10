#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define RESPONSE "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, World!"
#define SERVERPORT 8000

int main()
{
    // Create socket
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Failed to create socket");
        return 1;
    }

    // Prepare sockaddr_in structure
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(SERVERPORT);

    // Bind the socket
    if (bind(server_socket, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        perror("Bind failed");
        return 1;
    }

    // Listen for incoming connections
    listen(server_socket, 5);

    printf("Server listening on port %d...\n", SERVERPORT);

    while (1) {
        // Accept incoming connection
        int client_socket;
        struct sockaddr_in client_address;
        int client_address_length = sizeof(client_address);
        client_socket = accept(server_socket, (struct sockaddr *)&client_address, (socklen_t *)&client_address_length);
        if (client_socket < 0) {
            perror("Accept failed");
            return 1;
        }

        printf("Client connected: %s:%d\n", inet_ntoa(client_address.sin_addr), ntohs(client_address.sin_port));

        // Send response
        if (send(client_socket, RESPONSE, strlen(RESPONSE), 0) < 0) {
            perror("Send failed");
            return 1;
        }

        // Close the socket
        close(client_socket);
    }

    close(server_socket);

    return 0;
}
