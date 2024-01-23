# server for customized chatbot
import json
import os
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
APP_DATA_DB_CONN = None
KEY_DB_CONN = None


def create_table():
    # app data database
    cursor = APP_DATA_DB_CONN.cursor()
    # user tabble
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            monthly_budget INTEGER,
            status INTEGER,
            timestamp INTEGER,

            UNIQUE(user_id)
        );
        """
    )
    APP_DATA_DB_CONN.commit()

    # chat history table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            thread_id VARCHAR(255) UNIQUE,
            conversation TEXT,
            timestamp INTEGER,

            UNIQUE(thread_id)
        );
        """
    )
    APP_DATA_DB_CONN.commit()

    # saved prompt table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_prompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt_name VARCHAR(255),
            prompt_content TEXT,
            prompt_note TEXT,
            timestamp INTEGER,

            UNIQUE(user_id, prompt_name)
        );
        """
    )
    APP_DATA_DB_CONN.commit()

    # audit log table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            billing_period VARCHAR(255),
            thread_id VARCHAR(255),
            model VARCHAR(255),
            input_token INTEGER,
            cost_per_input_token DOUBLE,
            output_token INTEGER,
            cost_per_output_token DOUBLE,
            cost DOUBLE,
            timestamp INTEGER
        );
        """
    )
    APP_DATA_DB_CONN.commit()

    # audit status table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            billing_period VARCHAR(255),
            sum_cost DOUBLE,
            timestamp INTEGER
        );
        """
    )
    APP_DATA_DB_CONN.commit()

    # pricing table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pricing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model VARCHAR(255),
            cost_per_input_token DOUBLE,
            cost_per_output_token DOUBLE,
            start_time INTEGER,
            end_time INTEGER
        );
        """
    )
    APP_DATA_DB_CONN.commit()

    # key database
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_key (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(255),
            password_hash VARCHAR(255),
            status INTEGER,
            timestamp INTEGER,

            UNIQUE(username)
        );
        """
    )
    KEY_DB_CONN.commit()

    # api key database
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS api_key (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            api_key VARCHAR(255),
            timestamp INTEGER,

            UNIQUE(user_id, api_key)
        );
        """
    )
    KEY_DB_CONN.commit()


def get_chat_history(thread_id: str):
    cursor = APP_DATA_DB_CONN.cursor()
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
    cursor = APP_DATA_DB_CONN.cursor()

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
    APP_DATA_DB_CONN.commit()


def save_prompt(prompt_name: str, prompt_content: str, prompt_note: str):
    cursor = APP_DATA_DB_CONN.cursor()

    cursor.execute(
        """
        INSERT INTO saved_prompt (prompt_name, prompt_content, prompt_note, timestamp)
        VALUES (?, ?, ?, ?) 
        ON CONFLICT(prompt_name) DO UPDATE SET prompt_content = ?, timestamp = ?
        """,
        (
            prompt_name,
            prompt_content,
            prompt_note,
            int(time.time()),
            prompt_content,
            int(time.time()),
        ),
    )
    APP_DATA_DB_CONN.commit()


def delete_prompt(prompt_name: str):
    pass


def get_all_prompts():
    cursor = APP_DATA_DB_CONN.cursor()
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


def insert_user(username: str, password_hash: str):
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        INSERT INTO user_key (username, password_hash)
        VALUES (?, ?)
        """,
        (username, password_hash),
    )
    KEY_DB_CONN.commit()


def get_user(username: str):
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        SELECT username, password_hash FROM user_key WHERE username = ?
        """,
        (username,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    else:
        return {"username": row[0], "password_hash": row[1]}


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
        if not promptName or not promptContent:
            self.build_return(
                400, {"error": "promptName or promptContent is empty or not provided"}
            )
        else:
            save_prompt(promptName, promptContent, promptNote)
            print(f"Save prompt, name: {promptName}, content: {promptContent}")
            self.build_return(200)


class SignUpHandler(BaseHandler):
    def validate_passphrase(self, passphrase_hash):
        correct_passphrase_hash = os.environ.get("PASSPHRASE_HASH", None)
        if not correct_passphrase_hash:
            print("PASSPHRASE_HASH is not set")
            return False
        return passphrase_hash == correct_passphrase_hash

    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password_hash = body.get("password_hash")
        passphrase_hash = body.get("passphrase_hash")
        if not username or not password_hash or not passphrase_hash:
            self.build_return(
                400,
                {
                    "error": "username or password_hash or passphrase_hash is empty or not provided"
                },
            )
        elif not self.validate_passphrase(passphrase_hash):
            self.build_return(400, {"error": "passphrase is not correct"})
        else:
            try:
                insert_user(username, password_hash)
                self.build_return(200)
            except sqlite3.IntegrityError:
                self.build_return(400, {"error": "username already exists"})


class LogInHandler(BaseHandler):
    def validate_user(self, username, password_hash):
        user = get_user(username)
        if user is None:
            return False
        return user["password_hash"] == password_hash

    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password_hash = body.get("password_hash")
        if not username or not password_hash:
            self.build_return(
                400, {"error": "username or password_hash is empty or not provided"}
            )
        elif not self.validate_user(username, password_hash):
            self.build_return(
                400, {"error": "username or password_hash is not correct"}
            )
        else:
            self.build_return(200)


def make_app():
    return tornado.web.Application(
        [
            (r"/chat", ChatHandler),
            (r"/prompt", PromptHandler),
            (r"/login", LogInHandler),
            (r"/signup", SignUpHandler),
        ]
    )


if __name__ == "__main__":
    import sys

    app_data_db_name = str(sys.argv[1]) if len(sys.argv) > 1 else "test-app-data.db"
    key_db_name = str(sys.argv[2]) if len(sys.argv) > 2 else "test-key.db"

    APP_DATA_DB_CONN = sqlite3.connect(app_data_db_name)
    KEY_DB_CONN = sqlite3.connect(key_db_name)
    print("Connected to database: [%s] [%s]" % (app_data_db_name, key_db_name))

    create_table()
    app = make_app()
    app.listen(5600)
    print("Starting Tornado server on http://localhost:5600")
    tornado.ioloop.IOLoop.current().start()
