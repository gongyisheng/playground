# server for customized chatbot
import json
import uuid
import time

from openai import OpenAI
import tornado.ioloop
import tornado.web

OPENAI_CLIENT = OpenAI()
MESSAGE_STORAGE = {}
MODELS = None
MODELS_FETCH_TIME = 0

class BaseHandler(tornado.web.RequestHandler):
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

class ChatHandler(BaseHandler):
    def post(self):
        # get message from request body
        body = json.loads(self.request.body)
        system_message = body.get("systemMessage", "You are a helpful assistant")
        user_message = body.get("userMessage", "Repeat after me: I'm a helpful assistant")
        request_uuid = str(uuid.uuid4())
        MESSAGE_STORAGE[request_uuid] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        print(f"Receive post request, uuid: {request_uuid}")
        self.set_status(200)
        self.write({"uuid": request_uuid})
        self.finish()
    
    def build_sse_message(self, data):
        body = json.dumps({"content": data})
        return f"data: {body}\n\n"

    def get(self):
        global MESSAGE_STORAGE
        # set headers for SSE to work
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')

        # get uuid from url
        request_uuid = self.get_argument('uuid')
        model = self.get_argument('model')
        print(f"Receive get request, uuid: {request_uuid}, model: {model}")

        # create openai stream
        stream = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=MESSAGE_STORAGE[request_uuid],
            stream=True
        )
        assistant_message = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                assistant_message += content
                self.write(self.build_sse_message(content))
                self.flush()
        MESSAGE_STORAGE[request_uuid].append(
            {"role": "assistant", "content": assistant_message}
        )
        self.finish()

class ListModelsHandler(BaseHandler):
    def get(self):
        global MODELS, MODELS_FETCH_TIME
        if MODELS is None or time.time() - MODELS_FETCH_TIME > 3600:
            MODELS = []
            for model in OPENAI_CLIENT.models.list():
                model_id = str(model.id)
                if model_id.startswith(("gpt-3.5", "gpt-4")):
                    MODELS.append(model_id)
            MODELS = sorted(MODELS)
            MODELS_FETCH_TIME = time.time()

        self.set_status(200)
        self.write({"models": MODELS})
        self.finish()

def make_app():
    return tornado.web.Application([
        (r'/chat', ChatHandler),
        (r'/list_models', ListModelsHandler),
    ])

if __name__ == '__main__':
    app = make_app()
    app.listen(5600)
    print('Starting Tornado server on http://localhost:5600')
    tornado.ioloop.IOLoop.current().start()
