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

MODELS = ['gpt-3.5-turbo-1106','gpt-4-1106-preview']

# sqlite3 connection
# dev: test.db
# personal prompt engineering: prompt.db
# yipit: yipit.db
DB_CONN = None
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
            name VARCHAR(255),
            version VARCHAR(255),
            system_message TEXT,
            system_message_hash VARCHAR(32) UNIQUE,
            timestamp INTEGER,

            UNIQUE(name, version)
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

def get_chat_history(uuid: str):
    cursor = DB_CONN.cursor()
    cursor.execute(
        '''
        SELECT full_context FROM chat_history WHERE uuid = ?
        ''',
        (uuid,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    else:
        return json.loads(row[0])

def save_chat_history(uuid: str, model: str, full_context: list):
    cursor = DB_CONN.cursor()
    system_message = full_context[0]['content']
    system_message_hash = md5(system_message.encode('utf-8')).hexdigest()

    # insert chat history
    cursor.execute(
        '''
        INSERT INTO chat_history (uuid, model, system_message, system_message_hash, full_context, timestamp)
        VALUES (?, ?, ?, ?, ?, ?) 
        ON CONFLICT(uuid) DO UPDATE SET full_context = ?
        ''',
        (uuid, model, system_message, system_message_hash, json.dumps(full_context), int(time.time()), json.dumps(full_context))
    )
    DB_CONN.commit()

def save_prompt(name: str, version: str, system_message: str):
    cursor = DB_CONN.cursor()
    system_message_hash = md5(system_message.encode('utf-8')).hexdigest()

    cursor.execute(
        '''
        INSERT INTO saved_prompt (name, version, system_message, system_message_hash, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''',
        (name, version, system_message, system_message_hash, int(time.time()))
    )
    DB_CONN.commit()

def delete_prompt(name: str, version: str):
    pass

def get_all_prompts():
    cursor = DB_CONN.cursor()
    cursor.execute(
        '''
        SELECT name, version, system_message FROM saved_prompt
        '''
    )
    rows = cursor.fetchall()
    PROMPTS = {}
    for row in rows:
        name, version, system_message = row
        PROMPTS[f"{name}:{version}"] = {
            "name": name,
            "version": version,
            "systemMessage": system_message
        }
    return PROMPTS


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
    def validate_conversation(self, conversation):
        role = ["system", "user", "assistant"]
        for i in range(len(conversation)):
            item = conversation[i]
            if "role" not in item or "content" not in item:
                return False, "role or content is missing"
            correct_role = role[i%3] if i <= 2 else role[(i+1)%2+1]
            if item['role'] != correct_role:
                return False, "conversation role is not correct"
            if i == len(conversation) - 1 and item['role'] != 'user':
                return False, "last conversation role is not user"
        return True, None
    
    def post(self):
        # get message from request body
        body = json.loads(self.request.body)
        thread_id = body.get("thread_id", "")
        if thread_id == "":
            thread_id = str(uuid.uuid4())
        conversation = body.get("conversation", [])
        print(f"Receive post request, thread_id: {thread_id}, conversation: {conversation}")
        if len(conversation) == 0:
            conversation.append({"role": "system", "content": "You are a helpful assistant"})
        if len(conversation) == 1:
            conversation.append({"role": "user", "content": "Repeat after me: I'm a helpful assistant"})
        passed, error = self.validate_conversation(conversation)
        if not passed:
            self.set_status(400)
            self.write({"error": error})
            self.finish()
            return
        MESSAGE_STORAGE[thread_id] = conversation

        print(f"Receive post request, thread_id: {thread_id}, conversation: {conversation}")
        self.set_status(200)
        self.write({"thread_id": thread_id})
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
        thread_id = self.get_argument('thread_id')
        model = self.get_argument('model')
        conversation = MESSAGE_STORAGE[thread_id]
        print(f"Receive get request, uuid: {thread_id}, model: {model}")

        # create openai stream
        stream = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=conversation,
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
        conversation.append(
            {"role": "assistant", "content": assistant_message}
        )
        save_chat_history(thread_id, model, conversation)
        del MESSAGE_STORAGE[thread_id]
        self.finish()

class PromptHandler(BaseHandler):
    def get(self):
        prompts = get_all_prompts()
        self.write(json.dumps({"prompts": prompts}))
        self.set_status(200)
        self.finish()
    
    def post(self):
        body = json.loads(self.request.body)
        name = body.get("name", None)
        version = body.get("version", None)
        system_message = body.get("systemMessage", None)
        if name is None or system_message is None:
            self.set_status(400)
            self.finish()
        else:
            print(f"Save prompt, name: {name}, version: {version}")
            save_prompt(name, version, system_message)
            self.set_status(200)
            self.finish()

def make_app():
    return tornado.web.Application([
        (r'/chat', ChatHandler),
        (r'/prompt', PromptHandler)
    ])

if __name__ == '__main__':
    import sys
    db_name = str(sys.argv[1]) if len(sys.argv) > 1 else 'test.db'

    DB_CONN = sqlite3.connect(db_name)
    print("Connected to database: ", db_name)
    
    create_table()
    app = make_app()
    app.listen(5600)
    print('Starting Tornado server on http://localhost:5600')
    tornado.ioloop.IOLoop.current().start()
