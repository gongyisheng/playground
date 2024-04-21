import json

import tornado.ioloop
import tornado.web

from llama_cpp import Llama

# Put the location of to the GGUF model that you've download from HuggingFace here
MODEL_PATH = '/Users/temp/Downloads/llama-2-7b-chat.Q2_K.gguf'
MODEL = Llama(model_path=MODEL_PATH)

class ChatHandler(tornado.web.RequestHandler):

    def post(self):
        messages = json.loads(self.request.body)
        try:
            stream = MODEL.create_chat_completion_openai_v1(
                messages = messages,
                stream = True
            )

            for chunk in stream:
                print(chunk.choices[0].delta.content)

            self.set_status(200)
            return

        except Exception as e:
            self.set_status(500)
            self.write({"Error": str(e)})
            return

def make_app():
    return tornado.web.Application([
        (r"/llama", ChatHandler),
    ])

if __name__ == '__main__':
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()