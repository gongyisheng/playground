import json

import tornado.ioloop
import tornado.web

from llama_cpp import Llama

# Put the location of to the GGUF model that you've download from HuggingFace here
MODEL_PATH = '/Users/temp/Downloads/llama-2-7b-chat.Q2_K.gguf'
MODEL = Llama(model_path=MODEL_PATH)

def make_prompt(system_message, user_message):
    prompt = f"""<s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {user_message} [/INST]"""
    return prompt

class ChatHandler(tornado.web.RequestHandler):
    def post(self):
        body = json.loads(self.request.body)
        try:
            system_message = body.get('system_message')
            user_message = body.get('user_message')
            max_tokens = int(body.get('max_tokens', 4096))

            if system_message is None or user_message is None:
                self.set_status(400)
                self.write({"error": "Missing required parameters"})
                return
            else:
                # Prompt creation
                prompt = make_prompt(system_message, user_message)
                # Run the model
                output = MODEL(prompt, max_tokens=max_tokens, echo=True)

                self.set_status(200)
                self.write(output)
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