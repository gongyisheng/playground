# server for customized chatbot
from openai import OpenAI

# client = OpenAI()

# stream = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "Your name is John"},
#         {"role": "user", "content": "Repeat after me: My name is Alice"},
#     ],
#     stream=True
# )

# for chunk in stream:
#     content = chunk.choices[0].delta.content
#     if content is not None:
#         print(content)

import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, World!")

def make_app():
    return tornado.web.Application([
        (r'/', MainHandler),
    ])

if __name__ == '__main__':
    app = make_app()
    app.listen(5600)
    print('Starting Tornado server on http://localhost:5600')
    tornado.ioloop.IOLoop.current().start()
