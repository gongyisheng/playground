# server for customized chatbot
import json
import uuid
import sqlite3

from openai import OpenAI
import tornado.ioloop
import tornado.web

from models.user_model import UserModel
from models.session_model import SessionModel
from models.chat_history_model import ChatHistoryModel
from models.prompt_model import PromptModel
from models.audit_model import AuditModel

from models.api_key_model import ApiKeyModel
from models.user_key_model import UserKeyModel

MESSAGE_STORAGE = {}

MODEL_MAPPING = {
    "gpt-3.5": "gpt-3.5-turbo-1106",
    "gpt-4": "gpt-4-1106-preview",
}

SESSION_COOKIE_NAME = "_chat_session"
SESSION_COOKIE_EXPIRE_DAYS = 30 # 30 days
SESSION_COOKIE_PATH = "/"
# SESSION_COOKIE_DOMAIN = "yishenggong.com"

# sqlite3 connection
# dev: test.db
# personal prompt engineering: prompt.db
# yipit: yipit.db
class Global:
    api_key_model = None
    user_key_model = None
    user_model = None
    session_model = None
    chat_history_model = None
    prompt_model = None
    audit_model = None

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

class AuthHandler(BaseHandler):

    def prepare(self):
        # get session_id from cookie 
        session_id = self.get_cookie(SESSION_COOKIE_NAME)
        is_valid = Global.session_model.validate_session(session_id)
        if not is_valid:
            self.build_return(401, {"error": "Unauthorized operation"})
        else:
            self.user_id = Global.session_model.get_user_id_by_session(session_id)
            self.api_key = Global.api_key_model.get_api_key_by_user(self.user_id)

class ChatHandler(AuthHandler):

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
        OPENAI_CLIENT = OpenAI(api_key=self.api_key)
        stream = OPENAI_CLIENT.chat.completions.create(
            model=real_model, messages=conversation, stream=True, 
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
        Global.chat_history_model.save_chat_history(self.user_id, thread_id, conversation) # need user_id
        del MESSAGE_STORAGE[thread_id]
        self.finish()


class PromptHandler(AuthHandler):

    def get(self):
        print("Receive get request for prompts")
        my_prompts = {}
        for row in Global.prompt_model.get_prompts_by_user_id(self.user_id):
            prompt_name, prompt_content, prompt_note = row
            my_prompts[prompt_name] = {
                "promptName": prompt_name,
                "promptContent": prompt_content,
                "promptNote": prompt_note,
            }
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
            Global.prompt_model.save_prompt(self.user_id, promptName, promptContent, promptNote)
            print(f"Save prompt, name: {promptName}, content: {promptContent}")
            self.build_return(200)

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
        elif Global.user_key_model.validate_username(username):
            self.build_return(400, {"error": "Username already exists"})
        elif not Global.api_key_model.validate_invitation_code(invitation_code):
            self.build_return(
                400, {"error": "Invitation code is not correct or expired"}
            )
        else:
            user_id = Global.user_model.create_user()
            Global.user_key_model.create_user(user_id, username, password_hash)
            Global.api_key_model.claim_invitation_code(user_id, invitation_code)
            self.build_return(200)


class SignInHandler(BaseHandler):

    def post(self):
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
        elif not Global.user_key_model.validate_user(username, password_hash):
            self.build_return(400, {"error": "Username or password is not correct"})
        else:
            user_id = Global.user_key_model.get_user_id_by_username_password_hash(username, password_hash)
            session_id = Global.session_model.create_session(user_id)
            self.set_cookie(SESSION_COOKIE_NAME, session_id, expires_days=SESSION_COOKIE_EXPIRE_DAYS, path=SESSION_COOKIE_PATH)
            self.build_return(200)

class InviteHandler(BaseHandler):

    def post(self):
        body = json.loads(self.request.body)
        api_key = body.get("api_key")
        invitation_code = body.get("invitation_code")
        if not api_key:
            self.build_return(
                400,
                {"error": "API key is not provided"},
            )
        elif not invitation_code:
            self.build_return(
                400,
                {"error": "Invitation code is not provided"},
            )
        else:
            Global.api_key_model.insert_api_key_invitation_code(api_key, invitation_code)
            self.build_return(200)


def make_app():
    return tornado.web.Application(
        [
            (r"/chat", ChatHandler),
            (r"/prompt", PromptHandler),
            (r"/signin", SignInHandler),
            (r"/signup", SignUpHandler),
            (r"/invite", InviteHandler),
        ]
    )


if __name__ == "__main__":
    import sys

    app_data_db_name = str(sys.argv[1]) if len(sys.argv) > 1 else "test-app-data.db"
    key_db_name = str(sys.argv[2]) if len(sys.argv) > 2 else "test-key.db"

    APP_DATA_DB_CONN = sqlite3.connect(app_data_db_name)
    KEY_DB_CONN = sqlite3.connect(key_db_name)
    print("Connected to database: [%s] [%s]" % (app_data_db_name, key_db_name))

    Global.api_key_model = ApiKeyModel(KEY_DB_CONN)
    Global.api_key_model.create_tables()
    Global.user_key_model = UserKeyModel(KEY_DB_CONN)
    Global.user_key_model.create_tables()
    Global.user_model = UserModel(APP_DATA_DB_CONN)
    Global.user_model.create_tables()
    Global.session_model = SessionModel(APP_DATA_DB_CONN)
    Global.session_model.create_tables()
    Global.chat_history_model = ChatHistoryModel(APP_DATA_DB_CONN)
    Global.chat_history_model.create_tables()
    Global.prompt_model = PromptModel(APP_DATA_DB_CONN)
    Global.prompt_model.create_tables()
    Global.audit_model = AuditModel(APP_DATA_DB_CONN)
    Global.audit_model.create_tables()

    app = make_app()
    app.listen(5600)
    print("Starting Tornado server on http://localhost:5600")
    tornado.ioloop.IOLoop.current().start()
