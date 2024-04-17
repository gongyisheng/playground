import tornado.ioloop
import tornado.web
import random
import string

def get_random_string(length: int) -> str:
    # Generate a random string
    return ''.join(random.choices(string.ascii_letters, k=length))

class BaseHandler(tornado.web.RequestHandler):

    def set_default_headers(self) -> None:
        # Allow specific headers
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

        # Allow specific methods
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    
    def options(self):
        # Handle preflight OPTIONS requests
        self.set_status(204)
        self.finish()
    
    def get(self):
        length = random.randint(1, 1024*1024) # 1 MB
        self.write(get_random_string(length))
        self.set_status(200)
        self.finish()
    
    def post(self):
        length = random.randint(1, 1024*1024) # 1 MB
        self.write(get_random_string(length))
        self.set_status(200)
        self.finish()

def make_app():
    return tornado.web.Application([
        (r"/", BaseHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print('Server started at http://localhost:8888')
    tornado.ioloop.IOLoop.current().start()