# server for customized chatbot
import json
import time
import uuid

from openai import OpenAI
import tornado.ioloop
import tornado.web

OPENAI_CLIENT = OpenAI()
MESSAGE_STORAGE = {}
class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        # Allow all origins
        self.set_header("Access-Control-Allow-Origin", "*")
        
        # Allow specific headers
        self.set_header("Access-Control-Allow-Headers", "*")
        
        # Allow specific methods
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    
    def options(self):
        # Handle preflight OPTIONS requests
        self.set_status(204)
        self.finish()
    
    def post(self):
        # get message from request body
        body = json.loads(self.request.body)
        user_message = body.get("message", "Repeat after me: I'm a useful assistant")
        request_uuid = str(uuid.uuid4())
        MESSAGE_STORAGE[request_uuid] = user_message

        print(f"User message: {user_message}, uuid: {request_uuid}")
        self.set_status(200)
        self.write({"uuid": request_uuid})
        self.finish()
    
    def build_sse_message(self, data):
        return f"data: {data}\n\n"

    def get(self):
        # set headers for SSE to work
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')

        # get uuid from url
        request_uuid = self.get_argument('uuid')

        # create openai stream
        stream = OPENAI_CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a useful assistant"},
                {"role": "user", "content": MESSAGE_STORAGE[request_uuid]},
            ],
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                self.write(self.build_sse_message(content))
                self.flush()
        self.finish()

def make_app():
    return tornado.web.Application([
        (r'/', MainHandler),
    ])

if __name__ == '__main__':
    app = make_app()
    app.listen(5600)
    print('Starting Tornado server on http://localhost:5600')
    tornado.ioloop.IOLoop.current().start()
