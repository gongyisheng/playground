import random
from http.server import BaseHTTPRequestHandler, HTTPServer

options = ["猪", "圆", "胖", "球", "黄", "橘", "咪"]
length = 2

def randword():
    return "".join([options[random.randint(0, len(options) - 1)] for _ in range(length)])

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        text = ""
        for i in range(1000):
            text += randword()
            text += "\n"
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting http server on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()