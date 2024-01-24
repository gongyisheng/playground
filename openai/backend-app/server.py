# server for customized chatbot
import json
import uuid
import sqlite3
import time
import random
import string

from openai import OpenAI
import tornado.ioloop
import tornado.web

OPENAI_CLIENT = OpenAI()
MESSAGE_STORAGE = {}


SESSION_COOKIE_NAME = "_chat_session"
SESSION_COOKIE_EXPIRE_TIME = 60 * 60 * 24 * 30 # 30 days
SESSION_COOKIE_PATH = "/"
# SESSION_COOKIE_DOMAIN = "yishenggong.com"

# sqlite3 connection
# dev: test.db
# personal prompt engineering: prompt.db
# yipit: yipit.db
APP_DATA_DB_CONN = None
KEY_DB_CONN = None


def create_table():
    # app data database
    cursor = APP_DATA_DB_CONN.cursor()
    

    # key database
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_key (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username VARCHAR(255),
            password_hash VARCHAR(255),
            status INTEGER,
            timestamp INTEGER,

            UNIQUE(user_id)
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
            invitation_code VARCHAR(255),
            invitation_code_status INTEGER,
            timestamp INTEGER,

            UNIQUE(user_id)
            UNIQUE(api_key)
            UNIQUE(invitation_code)
        );
        """
    )
    KEY_DB_CONN.commit()









# def get_all_prompts():
#     cursor = APP_DATA_DB_CONN.cursor()
#     cursor.execute(
#         """
#         SELECT prompt_name, prompt_content, prompt_note FROM saved_prompt
#         """
#     )
#     rows = cursor.fetchall()
#     PROMPTS = {}
#     for row in rows:
#         prompt_name, prompt_content, prompt_note = row
#         PROMPTS[prompt_name] = {
#             "promptName": prompt_name,
#             "promptContent": prompt_content,
#             "promptNote": prompt_note,
#         }
#     return PROMPTS


def create_user(username: str, password_hash: str):
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        INSERT INTO user_key (username, password_hash)
        VALUES (?, ?)
        """,
        (username, password_hash),
    )
    KEY_DB_CONN.commit()


def get_user_id_by_username(username: str):
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        SELECT user_id FROM user_key WHERE username = ?
        """,
        (username,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    else:
        return row[0]

def validate_user(username: str, password_hash: str):
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
        return row[1] == password_hash


def validate_invitation_code(invitation_code: str):
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        SELECT invitation_code_status FROM api_key WHERE invitation_code = ?
        """,
        (invitation_code,),
    )
    row = cursor.fetchone()
    if row is None:
        return False
    else:
        return row[0] == INVITATION_CODE_STATUS_ACTIVE


def expire_invitation_code(invitation_code: str):
    cursor = KEY_DB_CONN.cursor()
    cursor.execute(
        """
        UPDATE api_key SET invitation_code_status = ? WHERE invitation_code = ?
        """,
        (INVITATION_CODE_STATUS_EXPIRED, invitation_code),
    )
    KEY_DB_CONN.commit()


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


class SessionManager():
    def __init__(self):
        self.cursor = APP_DATA_DB_CONN.cursor()


        

    def get_user_id_by_session(self, session_id: str):
        self.cursor.execute(
            """
            SELECT user_id FROM session WHERE session_id = ?
            """,
            (session_id,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        else:
            return row[0]


class SignUpHandler(BaseHandler):
    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password_hash = body.get("password_hash")
        invitation_code = body.get("invitation_code")
        if not username:
            self.build_return(
                400,
                {"error": "Username is not provided"},
            )
        elif not password_hash:
            self.build_return(
                400,
                {"error": "Password is not provided"},
            )
        elif not invitation_code:
            self.build_return(
                400,
                {"error": "Invitation code is not provided"},
            )
        elif not validate_invitation_code(invitation_code):
            self.build_return(
                400, {"error": "Invitation code is not correct or expired"}
            )
        else:
            try:
                expire_invitation_code(invitation_code)
                create_user(username, password_hash)
                self.build_return(200)
            except sqlite3.IntegrityError:
                self.build_return(400, {"error": "Username already exists"})


class SignInHandler(BaseHandler):

    session_manager = SessionManager()

    def post(self):
        cookie = self.get_cookie("_chat_session")
        if cookie is not None:
            print(f"Receive post request, cookie: {cookie}")
            _, session_id = cookie.split("=")
            if self.session_manager.validate_session(user_id, session_id):
                self.build_return(200)
        else:
            body = json.loads(self.request.body)
            username = body.get("username")
            password_hash = body.get("password_hash")
            print(f"Receive post request, username: {username}")
            if not username:
                self.build_return(
                    400, {"error": "Username is not provided"}
                )
            elif not password_hash:
                self.build_return(
                    400, {"error": "Password is not provided"}
                )
            elif not validate_user(username, password_hash):
                self.build_return(400, {"error": "Username or password is not correct"})
            else:
                user_id = get_user_id_by_username(username)
                session_id = create_session(user_id)
                self.set_cookie("_chat_session", session_id)


def make_app():
    return tornado.web.Application(
        [
            (r"/chat", ChatHandler),
            (r"/prompt", PromptHandler),
            (r"/signin", SignInHandler),
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
