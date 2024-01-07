# server for customized chatbot
from hashlib import md5
import json
import uuid
import sqlite3
import time

from openai import OpenAI
import tornado.ioloop
import tornado.web

OPENAI_CLIENT = OpenAI()
MESSAGE_STORAGE = {}

MODELS = None
MODELS_FETCH_TIME = 0

# sqlite3 connection
# dev: test.db
# prompt engineering: prompt.db
DB_CONN = sqlite3.connect('test.db')

def create_table():
    cursor = DB_CONN.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid VARCHAR(255) UNIQUE,
            model VARCHAR(255),
            system_message TEXT,
            system_message_hash VARCHAR(32),
            full_context TEXT,
            timestamp INTEGER
        );
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_uuid ON chat_history (uuid);
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_system_message_hash ON chat_history (system_message_hash);
        '''
    )
    DB_CONN.commit()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS saved_prompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) UNIQUE,
            version INTEGER,
            system_message TEXT,
            system_message_hash VARCHAR(32),
            insert_time INTEGER,
            update_time INTEGER
        );
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_name ON saved_prompt (name);
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_system_message_hash ON saved_prompt (system_message_hash);
        '''
    )
    DB_CONN.commit()

def insert_chat_history(uuid: str, model: str, full_context: list):
    cursor = DB_CONN.cursor()
    system_message = full_context[0]['content']
    system_message_hash = md5(system_message.encode('utf-8')).hexdigest()

    # insert chat history
    cursor.execute(
        '''
        INSERT INTO chat_history (uuid, model, system_message, system_message_hash, full_context, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''',
        (uuid, model, system_message, system_message_hash, json.dumps(full_context), int(time.time()))
    )
    DB_CONN.commit()

def save_prompt(name: str, system_message: str):
    cursor = DB_CONN.cursor()
    system_message_hash = md5(system_message.encode('utf-8')).hexdigest()

    # check if system message exists
    cursor.execute(
        '''
        SELECT id, version FROM saved_prompt
        WHERE name = ?
        ''',
        (name,)
    )
    row = cursor.fetchone()

    if row is None:
        cursor.execute(
            '''
            INSERT INTO saved_prompt (name, version, system_message, system_message_hash, insert_time, update_time)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (name, 1, system_message, system_message_hash, int(time.time()), int(time.time()))
        )
    else:
        id, version = row
        cursor.execute(
            '''
            UPDATE saved_prompt
            SET version = ?, system_message = ?, system_message_hash = ?, update_time = ?
            WHERE id = ?
            ''',
            (version + 1, system_message, system_message_hash, int(time.time()), id)
        )
    DB_CONN.commit()

def get_all_prompts():
    cursor = DB_CONN.cursor()
    cursor.execute(
        '''
        SELECT name, system_message FROM saved_prompt
        '''
    )
    rows = cursor.fetchall()
    return rows


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
        full_context = MESSAGE_STORAGE[request_uuid]
        print(f"Receive get request, uuid: {request_uuid}, model: {model}")

        # create openai stream
        stream = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=full_context,
            stream=True
        )

        assistant_message = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                assistant_message += content
                self.write(self.build_sse_message(content))
                self.flush()

        # save chat history
        full_context.append(
            {"role": "assistant", "content": assistant_message}
        )
        insert_chat_history(request_uuid, model, full_context)
        del MESSAGE_STORAGE[request_uuid]
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

class PromptHandler(BaseHandler):
    def get(self):
        prompts = get_all_prompts()
        body = []
        for prompt in prompts:
            name, system_message = prompt
            body.append({
                "name": name,
                "systemMessage": system_message
            })
        self.write({"prompts": body})
        self.set_status(200)
        self.finish()
    
    def post(self):
        body = json.loads(self.request.body)
        name = body.get("name", None)
        system_message = body.get("systemMessage", None)
        if name is None or system_message is None:
            self.set_status(400)
            self.finish()
        else:
            save_prompt(name, system_message)
            self.set_status(200)
            self.finish()

def make_app():
    return tornado.web.Application([
        (r'/chat', ChatHandler),
        (r'/list_models', ListModelsHandler),
        (r'/prompt', PromptHandler)
    ])

if __name__ == '__main__':
    create_table()
    app = make_app()
    app.listen(5600)
    print('Starting Tornado server on http://localhost:5600')
    tornado.ioloop.IOLoop.current().start()
