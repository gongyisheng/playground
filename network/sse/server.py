import http.server
import socketserver
import time

# Custom request handler for SSE
# Headers are important for SSE to work
    # Content-Type: text/event-stream
    # Cache-Control: no-cache
    # Connection: keep-alive
class SSEHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/sse':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            count = 0
            while True:
                time.sleep(1)
                count += 1
                message = f"data: {count}\n"
                self.wfile.write(message.encode('utf-8'))
                self.wfile.flush()
        else:
            super().do_GET()

if __name__ == '__main__':
    PORT = 8000
    with socketserver.TCPServer(("", PORT), SSEHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()