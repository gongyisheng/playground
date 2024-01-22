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

MODEL_MAPPING = {
    "gpt-3.5": "gpt-3.5-turbo-1106",
    "gpt-4": "gpt-4-1106-preview",
}

# sqlite3 connection
# dev: test.db
# personal prompt engineering: prompt.db
# yipit: yipit.db
DB_CONN = None


def create_table():
    cursor = DB_CONN.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id VARCHAR(255) UNIQUE,
            conversation TEXT,
            timestamp INTEGER,

            UNIQUE(thread_id)
        );
        """
    )
    DB_CONN.commit()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_prompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_name VARCHAR(255),
            prompt_content TEXT,
            prompt_note TEXT,
            timestamp INTEGER,

            UNIQUE(prompt_name)
        );
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_name ON saved_prompt (prompt_name);
        """
    )
    DB_CONN.commit()


def get_chat_history(thread_id: str):
    cursor = DB_CONN.cursor()
    cursor.execute(
        """
        SELECT conversation FROM chat_history WHERE thread_id = ?
        """,
        (thread_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    else:
        return json.loads(row[0])


def save_chat_history(thread_id: str, conversation: list):
    cursor = DB_CONN.cursor()

    # insert chat history
    cursor.execute(
        """
        INSERT INTO chat_history (thread_id, conversation, timestamp)
        VALUES (?, ?, ?) 
        ON CONFLICT(thread_id) DO UPDATE SET conversation = ?
        """,
        (
            thread_id,
            json.dumps(conversation),
            int(time.time()),
            json.dumps(conversation),
        ),
    )
    DB_CONN.commit()


def save_prompt(prompt_name: str, prompt_content: str, prompt_note: str):
    cursor = DB_CONN.cursor()

    cursor.execute(
        """
        INSERT INTO saved_prompt (prompt_name, prompt_content, prompt_note, timestamp)
        VALUES (?, ?, ?, ?) 
        ON CONFLICT(prompt_name) DO UPDATE SET prompt_content = ?
        """,
        (prompt_name, prompt_content, prompt_note, int(time.time()), prompt_content),
    )
    DB_CONN.commit()


def delete_prompt(prompt_name: str):
    pass


def get_all_prompts():
    cursor = DB_CONN.cursor()
    cursor.execute(
        """
        SELECT prompt_name, prompt_content, prompt_note FROM saved_prompt
        """
    )
    rows = cursor.fetchall()
    PROMPTS = {}
    for row in rows:
        prompt_name, prompt_content, prompt_note = row
        PROMPTS[prompt_name] = {
            "promptName": prompt_name,
            "promptContent": prompt_content,
            "promptNote": prompt_note,
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

    def build_return(self, status_code, message=None):
        self.set_status(status_code)
        if message is not None:
            self.write(message)
        self.finish()
        return


class ChatHandler(BaseHandler):
    def validate_conversation(self, conversation):
        role = ["system", "user", "assistant"]
        for i in range(len(conversation)):
            item = conversation[i]
            if "role" not in item or "content" not in item:
                return False, "role or content is missing"
            correct_role = role[i % 3] if i <= 2 else role[(i + 1) % 2 + 1]
            if item["role"] != correct_role:
                return False, "conversation role is not correct"
            if i == len(conversation) - 1 and item["role"] != "user":
                return False, "last conversation role is not user"
        return True, None

    def post(self):
        # get message from request body
        body = json.loads(self.request.body)
        thread_id = body.get("thread_id", "")
        if thread_id == "":
            thread_id = str(uuid.uuid4())
        conversation = body.get("conversation", [])
        print(
            f"Receive post request, thread_id: {thread_id}, conversation: {conversation}"
        )
        if len(conversation) == 0:
            conversation.append(
                {"role": "system", "content": "You are a helpful assistant"}
            )
        if len(conversation) == 1:
            conversation.append(
                {"role": "user", "content": "Repeat after me: I'm a helpful assistant"}
            )
        passed, error = self.validate_conversation(conversation)
        if not passed:
            self.build_return(400, {"error": error})
        MESSAGE_STORAGE[thread_id] = conversation

        print(
            f"Receive post request, thread_id: {thread_id}, conversation: {conversation}"
        )
        self.build_return(200, {"thread_id": thread_id})

    def build_sse_message(self, data):
        body = json.dumps({"content": data})
        return f"data: {body}\n\n"

    def get(self):
        global MESSAGE_STORAGE
        # set headers for SSE to work
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Connection", "keep-alive")

        # get uuid from url
        thread_id = self.get_argument("thread_id")
        model = self.get_argument("model")
        if thread_id is None or model is None:
            self.build_return(400, {"error": "thread_id or model is missing"})

        real_model = MODEL_MAPPING.get(model.lower(), None)
        if real_model is None:
            self.build_return(400, {"error": "model is not supported"})

        conversation = MESSAGE_STORAGE.get(thread_id)
        if conversation is None:
            self.build_return(400, {"error": "thread_id not found in message storage"})
        print(f"Receive get request, uuid: {thread_id}, model: {real_model}")

        # create openai stream
        stream = OPENAI_CLIENT.chat.completions.create(
            model=real_model, messages=conversation, stream=True
        )

        assistant_message = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                assistant_message += content
                self.write(self.build_sse_message(content))
                self.flush()

        # save chat history
        conversation.append({"role": "assistant", "content": assistant_message})
        save_chat_history(thread_id, conversation)
        del MESSAGE_STORAGE[thread_id]
        self.finish()


class PromptHandler(BaseHandler):
    def get(self):
        print("Receive get request for prompts")
        my_prompts = get_all_prompts()
        self.write(json.dumps(my_prompts))
        self.set_status(200)
        self.finish()

    def post(self):
        body = json.loads(self.request.body)
        promptName = body.get("promptName", None)
        promptContent = body.get("promptContent", None)
        promptNote = body.get("promptNote", "")
        if promptName is None or promptContent is None:
            self.build_return(400, {"error": "promptName or promptContent is missing"})
        else:
            print(f"Save prompt, name: {promptName}, content: {promptContent}")
            save_prompt(promptName, promptContent, promptNote)
            self.build_return(200)


def make_app():
    return tornado.web.Application(
        [(r"/chat", ChatHandler), (r"/prompt", PromptHandler)]
    )


if __name__ == "__main__":
    import sys

    db_name = str(sys.argv[1]) if len(sys.argv) > 1 else "test.db"

    DB_CONN = sqlite3.connect(db_name)
    print("Connected to database: ", db_name)

    create_table()
    app = make_app()
    app.listen(5600)
    print("Starting Tornado server on http://localhost:5600")
    tornado.ioloop.IOLoop.current().start()
