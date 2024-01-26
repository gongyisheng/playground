# server for customized chatbot
from datetime import datetime
import json
import sqlite3
import uuid

from Crypto.Cipher import AES
from openai import OpenAI
import yaml
import tiktoken
import tiktoken_ext.openai_public
import tornado.ioloop
import tornado.web

from models.user_model import UserModel
from models.session_model import SessionModel
from models.chat_history_model import ChatHistoryModel
from models.prompt_model import PromptModel
from models.audit_model import AuditModel
from models.pricing_model import PricingModel

from models.api_key_model import ApiKeyModel
from models.user_key_model import UserKeyModel

MESSAGE_STORAGE = {}
ENC = tiktoken.core.Encoding(**tiktoken_ext.openai_public.cl100k_base())


# read yaml config file
def read_config(file_path):
    with open(file_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read config successful. env: %s" % data["environment"])
    return data


class Global:
    api_key_model = None
    user_key_model = None
    user_model = None
    session_model = None
    chat_history_model = None
    prompt_model = None
    audit_model = None
    pricing_model = None
    config = None


# encrypt and decrypt utils functions
def encrypt_data(plaintext: str, key: str, salt: str) -> bytes:
    cipher = AES.new(key.encode("ascii"), AES.MODE_CFB, iv=salt.encode("ascii"))
    ciphertext = cipher.encrypt(plaintext.encode("ascii"))
    return ciphertext


def decrypt_data(ciphertext: bytes, key: str, salt: str) -> str:
    cipher = AES.new(key.encode("ascii"), AES.MODE_CFB, iv=salt.encode("ascii"))
    plaintext = cipher.decrypt(ciphertext).decode("ascii")
    return plaintext


# base handler for all requests, cors configs
class BaseHandler(tornado.web.RequestHandler):
    def prepare(self):
        super().prepare()
        self.encrypt_key = Global.config["encrypt_key"]
        self.encrypt_salt = Global.config["encrypt_salt"]

    def set_default_headers(self):
        # Only allow origins from nginx reverse proxy
        self.set_header(
            "Access-Control-Allow-Origin", Global.config["access_control_allow_origin"]
        )

        # Allow specific headers
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

        # Allow specific methods
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        # Allow credentials
        self.set_header("Access-Control-Allow-Credentials", "true")

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


# handler for all authenticated requests, session is get from cookie
class AuthHandler(BaseHandler):
    def prepare(self):
        super().prepare()
        # options request does not need auth
        if self.request.method == "OPTIONS":
            return

        # get session_id from cookie
        session_id = self.get_cookie(Global.config["session_cookie_name"], None)
        is_valid = Global.session_model.validate_session(session_id)
        if not is_valid:
            self.build_return(401, {"error": "Unauthorized operation"})
            return
        else:
            self.user_id = Global.session_model.get_user_id_by_session(session_id)
            encrypted_api_key = Global.api_key_model.get_api_key_by_user(self.user_id)
            self.api_key = decrypt_data(
                encrypted_api_key, self.encrypt_key, self.encrypt_salt
            )


# handler for chat request (POST and GET)
# POST: save prompt and context to memory storage, validation
# GET:  send prompt to openai and transfer completion SSE stream to client, audit, save chat history
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
            return
        else:
            MESSAGE_STORAGE[thread_id] = conversation
            print(
                f"Receive post request, thread_id: {thread_id}, conversation: {conversation}"
            )
            self.build_return(200, {"thread_id": thread_id})
            return

    def build_sse_message(self, data):
        body = json.dumps({"content": data})
        return f"data: {body}\n\n"

    def get_token_num(self, message):
        num_tokens = len(ENC.encode(message))
        return num_tokens

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
            return

        real_model = Global.config["model_mapping"].get(model.lower(), None)
        if real_model is None:
            self.build_return(400, {"error": "model is not supported"})
            return

        conversation = MESSAGE_STORAGE.get(thread_id)
        if conversation is None:
            self.build_return(400, {"error": "thread_id not found in message storage"})
            return
        print(f"Receive get request, uuid: {thread_id}, model: {real_model}")

        billing_period = datetime.now().strftime("%Y-%m")
        user_cost = Global.audit_model.get_total_cost_by_user_id_billing_period(
            self.user_id, billing_period
        )
        user_limit = Global.audit_model.get_budget_by_user_id(self.user_id)

        if user_cost >= user_limit:
            del MESSAGE_STORAGE[thread_id]
            self.write(
                self.build_sse_message(
                    "You have exceeded your monthly budget. Please contact to author.\nhttps://yishenggong.com/about-me"
                )
            )
            self.flush()
            return

        # create openai stream
        OPENAI_CLIENT = OpenAI(api_key=self.api_key)
        stream = OPENAI_CLIENT.chat.completions.create(
            model=real_model,
            messages=conversation,
            stream=True,
        )

        assistant_message = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                assistant_message += content
                self.write(self.build_sse_message(content))
                self.flush()

        del MESSAGE_STORAGE[thread_id]

        # save chat history
        conversation.append({"role": "assistant", "content": assistant_message})
        Global.chat_history_model.save_chat_history(
            self.user_id, thread_id, conversation
        )

        # audit cost
        input_token = 0
        for i in range(len(conversation) - 1):
            item = conversation[i]
            input_token += self.get_token_num(item["content"])
        print(f"input_token: {input_token}")

        output_token = self.get_token_num(assistant_message)
        print(f"output_token: {output_token}")

        billing_period = datetime.now().strftime("%Y-%m")
        Global.audit_model.insert_audit_log(
            self.user_id,
            billing_period,
            thread_id,
            real_model,
            input_token,
            output_token,
        )

        self.finish()


# handler for prompt request (POST and GET)
# POST: insert/update prompt to database, validation
# GET:  get prompt from database
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
            return
        else:
            Global.prompt_model.save_prompt(
                self.user_id, promptName, promptContent, promptNote
            )
            print(f"Save prompt, name: {promptName}, content: {promptContent}")
            self.build_return(200)
            return


# handler for signup request (POST)
# POST: insert user to database, validation with invitation code, encrypt password
class SignUpHandler(BaseHandler):
    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password = body.get("password")
        invitation_code = body.get("invitation_code")
        if not username:
            self.build_return(
                400,
                {"error": "Username is not provided"},
            )
            return
        elif not password:
            self.build_return(
                400,
                {"error": "Password is not provided"},
            )
            return
        elif not invitation_code:
            self.build_return(
                400,
                {"error": "Invitation code is not provided"},
            )
            return
        elif Global.user_key_model.validate_username(username):
            self.build_return(400, {"error": "Username already exists"})
            return

        encrypted_invitation_code = encrypt_data(
            invitation_code, self.encrypt_key, self.encrypt_salt
        )
        if not Global.api_key_model.validate_invitation_code(encrypted_invitation_code):
            self.build_return(
                400, {"error": "Invitation code is not correct or expired"}
            )
            return
        else:
            user_id = Global.user_model.create_user()
            encrypted_pwd = encrypt_data(password, self.encrypt_key, self.encrypt_salt)
            Global.user_key_model.create_user(user_id, username, encrypted_pwd)
            Global.api_key_model.claim_invitation_code(
                user_id, encrypted_invitation_code
            )
            Global.audit_model.insert_budget_by_user_id(
                user_id, Global.config["default_monthly_budget"]
            )
            self.build_return(200)
            return


# handler for signin request (POST)
# POST: validate user, create session, set cookie
class SignInHandler(BaseHandler):
    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password = body.get("password")
        print(f"Receive post request, username: {username}")
        if not username:
            self.build_return(400, {"error": "Username is not provided"})
            return
        elif not password:
            self.build_return(400, {"error": "Password is not provided"})
            return

        encrypted_pwd = encrypt_data(password, self.encrypt_key, self.encrypt_salt)
        if not Global.user_key_model.validate_user(username, encrypted_pwd):
            self.build_return(400, {"error": "Username or password is not correct"})
            return
        else:
            user_id = Global.user_key_model.get_user_id_by_username_password(
                username, encrypted_pwd
            )
            session_id = Global.session_model.create_session(user_id)
            self.set_cookie(
                Global.config["session_cookie_name"],
                session_id,
                expires_days=Global.config["session_cookie_expires_days"],
                path=Global.config["session_cookie_path"],
                domain=Global.config.get("session_cookie_domain"),
            )
            self.build_return(200)
            return


# handler for invite request (POST)
# POST: insert api key and invitation code to database, validation, encrypt api key and invitation code before insert
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
            return
        elif not invitation_code:
            self.build_return(
                400,
                {"error": "Invitation code is not provided"},
            )
            return
        else:
            encrypted_api_key = encrypt_data(
                api_key, self.encrypt_key, self.encrypt_salt
            )
            encrypted_invitation_code = encrypt_data(
                invitation_code, self.encrypt_key, self.encrypt_salt
            )
            Global.api_key_model.insert_api_key_invitation_code(
                encrypted_api_key, encrypted_invitation_code
            )
            self.build_return(200)
            return


# handler for audit request (GET)
# GET: get total cost and budget of current user from database
class AuditHandler(AuthHandler):
    def get(self):
        print("Receive get request for audit")
        billing_period = datetime.now().strftime("%Y-%m")
        cost = Global.audit_model.get_total_cost_by_user_id_billing_period(
            self.user_id, billing_period
        )
        limit = Global.audit_model.get_budget_by_user_id(self.user_id)
        self.build_return(
            200,
            {
                "cost": str(int(cost * 10000) / 10000.0),
                "limit": str(int(limit * 100) / 100.0),
            },
        )
        return


def make_app():
    return tornado.web.Application(
        [
            (r"/chat", ChatHandler),
            (r"/prompt", PromptHandler),
            (r"/signin", SignInHandler),
            (r"/signup", SignUpHandler),
            (r"/invite", InviteHandler),
            (r"/audit", AuditHandler),
        ]
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", dest="config", default="backend-config.test.yaml")
    args = parser.parse_args()

    Global.config = read_config(args.config)

    APP_DATA_DB_CONN = sqlite3.connect(Global.config["app_data_db_name"])
    KEY_DB_CONN = sqlite3.connect(Global.config["key_db_name"])
    print(
        "Connected to database: [%s] [%s]"
        % (Global.config["app_data_db_name"], Global.config["key_db_name"])
    )

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
    Global.pricing_model = PricingModel(APP_DATA_DB_CONN)
    Global.pricing_model.create_tables()

    # init price
    if Global.pricing_model.get_current_pricing_by_model("gpt-3.5-turbo-1106") is None:
        Global.pricing_model.create_pricing(
            "gpt-3.5-turbo-1106", 0.001 / 1000, 0.002 / 1000, 0
        )
    if Global.pricing_model.get_current_pricing_by_model("gpt-4-1106-preview") is None:
        Global.pricing_model.create_pricing(
            "gpt-4-1106-preview", 0.01 / 1000, 0.03 / 1000, 0
        )

    app = make_app()
    if Global.config["enable_https"]:
        app.listen(
            Global.config["port"],
            ssl_options={
                "certfile": Global.config["https_certfile_path"],
                "keyfile": Global.config["https_keyfile_path"],
            },
        )
    else:
        app.listen(Global.config["port"])
    print(f"Starting Tornado server on localhost:{Global.config['port']}")

    tornado.ioloop.IOLoop.current().start()
