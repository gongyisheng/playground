# server for customized chatbot
import base64
from datetime import datetime
import json
import logging
import os
import random
import sqlite3
import string
import sys
import uuid

sys.path.append("../")

from openai import AsyncOpenAI
import anthropic
import yaml
import tornado.ioloop
import tornado.web

from models.user_model import UserModel
from models.session_model import SessionModel
from models.chat_history_model import ChatHistoryModel
from models.prompt_model import PromptModel
from models.audit_model import AuditModel
from models.pricing_model import PricingModel

from models.invitation_code_model import InvitationCodeModel
from models.user_key_model import UserKeyModel

from one_time.init_price import init_price

# from one_time.init_prompt import init_default_prompt

from utils import setup_logger, encrypt_data, decrypt_data

MESSAGE_STORAGE = {}


# read yaml config file
def read_config(file_path):
    with open(file_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        logging.debug(f"Read config successful. env: {data['environment']}")
    return data


class Global:
    user_key_model = None
    invitation_code_model = None
    user_model = None
    session_model = None
    chat_history_model = None
    prompt_model = None
    audit_model = None
    pricing_model = None
    config = None
    openai_client = None
    claude_client = None
    gemini_client = None


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


class PingHandler(BaseHandler):
    def get(self):
        self.write("pong")
        self.set_status(200)
        self.finish()


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

    def get(self):
        self.build_return(200)

# handler for file request (POST and GET)
# POST: save file to database and local disk
# GET:  get file list of a user from database
class FileHandler(AuthHandler):
    def post(self):
        thread_id = thread_id = self.get_argument("thread_id", None)
        if thread_id is None or thread_id == "":
            self.build_return(400, {"error": "thread_id is missing"})
            return
        file_infos = self.request.files['files']

        for file_info in file_infos:
            file_name = file_info['filename']
            file_body = file_info['body']
            
            base64_string = base64.b64encode(file_body).decode("utf-8")
            file_size = len(base64_string) / 1024 / 1024 # MB

            path = os.path.join(Global.config['user_file_upload_dir'], str(self.user_id), str(thread_id), file_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(file_body)
            logging.info(f"Upload file: {file_name}, user_id: {self.user_id}, thread_id: {thread_id}, size: {round(file_size, 2)} MB")

        self.build_return(200)

    def delete(self):
        body = json.loads(self.request.body)
        thread_id = body.get("thread_id", None)
        if thread_id is None or thread_id == "":
            self.build_return(400, {"error": "thread_id is missing"})
            return
        file_name = body.get("file_name")
        if file_name is None or file_name == "":
            self.build_return(400, {"error": "file_name is missing"})
            return
        
        file_path = os.path.join(Global.config['user_file_upload_dir'], str(self.user_id), str(thread_id), file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Delete file: {file_name}, user_id: {self.user_id}, thread_id: {thread_id}")
        else:
            logging.info(f"Try to delete but file not found: {file_name}, user_id: {self.user_id}, thread_id: {thread_id}")
        self.build_return(200)


# handler for chat request (POST and GET)
# POST: save prompt and context to memory storage, validation
# GET:  send prompt to openai and transfer completion SSE stream to client, audit, save chat history
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
class ChatHandler(AuthHandler):
    def validate_conversation(self, conversation, thread_id):
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
            if "files" in item:
                for file_name in item["files"]:
                    if not os.path.exists(os.path.join(Global.config['user_file_upload_dir'], str(self.user_id), str(thread_id), file_name)):
                        return False, f"file {file_name} not found"
        return True, None

    def post(self):
        # get message from request body
        body = json.loads(self.request.body)
        thread_id = body.get("thread_id", None)
        if thread_id is None or thread_id == "":
            self.build_return(400, {"error": "thread_id is missing"})
            return

        conversation = body.get("conversation", [])
        if len(conversation) == 0:
            conversation.append(
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
            )
        passed, error = self.validate_conversation(conversation, thread_id)
        if not passed:
            self.build_return(400, {"error": error})
            return

        for i in range(len(conversation)):
            item = conversation[i]
            text_content = item["content"]
            if "files" in item:
                new_content = []
                for file_name in item["files"]:
                    file_path = os.path.join(Global.config['user_file_upload_dir'], str(self.user_id), str(thread_id), file_name)
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    base64_string = base64.b64encode(file_content).decode("utf-8")
                    new_content.append({
                        "type": "file",
                        "file": {
                            "filename": file_name,
                            "file_data": f"data:application/pdf;base64,{base64_string}"
                        }
                    })
                new_content.append({
                    "type": "text",
                    "text": text_content
                })
                conversation[i]["content"] = new_content
                conversation[i].pop("files")
            else:
                pass

        # save conversation to memory storage
        MESSAGE_STORAGE[thread_id] = conversation
        self.build_return(200, {"thread_id": thread_id})
        return

    def build_sse_message(self, data):
        body = json.dumps({"content": data})
        return f"data: {body}\n\n"

    async def openai_text_stream(self, real_model, conversation):
        if Global.openai_client is None:
            Global.openai_client = AsyncOpenAI(api_key=Global.config["openai_api_key"])

        client  = Global.openai_client
        stream = await client.chat.completions.create(
            model=real_model,
            messages=conversation,
            stream=True,
            stream_options={"include_usage": True}
        )
        async for chunk in stream:
            if len(chunk.choices) == 0:
                yield {
                    "type": "usage", 
                    "content": {
                        "input_tokens": chunk.usage.prompt_tokens, 
                        "output_tokens": chunk.usage.completion_tokens
                        }
                    }
            else:
                yield {
                    "type": "text",
                    "content": chunk.choices[0].delta.content
                }

    async def gemini_text_stream(self, real_model, conversation):
        if Global.gemini_client is None:
            Global.gemini_client = AsyncOpenAI(
                api_key=Global.config["gemini_api_key"],
                base_url=Global.config["gemini_base_url"],
            )
        
        client = Global.gemini_client
        stream = await client.chat.completions.create(
            model=real_model,
            messages=conversation,
            stream=True,
            stream_options={
                "include_usage": True
            }
        )
        async for chunk in stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                logging.info(chunk)
                yield {
                    "type": "usage", 
                    "content": {
                        "input_tokens": chunk.usage.prompt_tokens, 
                        "output_tokens": chunk.usage.completion_tokens
                        }
                    }
            elif chunk.choices[0].delta.content:
                yield {
                    "type": "text",
                    "content": chunk.choices[0].delta.content
                }

    async def claude_text_stream(self, real_model, conversation):
        system_message = conversation[0]["content"]
        conversation = conversation[1:]

        if Global.claude_client is None:
            Global.claude_client = anthropic.AsyncAnthropic(api_key=Global.config["claude_api_key"])
        
        client = Global.claude_client
        async with client.messages.stream(
            max_tokens=8192,
            messages=conversation,
            model=real_model,
            system=system_message,
        ) as stream:
            async for text in stream.text_stream:
                yield {
                    "type": "text",
                    "content": text
                }
            # logging.info(stream._AsyncMessageStream__final_message_snapshot)
            final_msg = await stream.get_final_message()
            yield {
                "type": "usage",
                "content": {
                    "input_tokens": final_msg.usage.input_tokens,
                    "output_tokens": final_msg.usage.output_tokens,
                }
            }


    async def get(self):
        global MESSAGE_STORAGE
        # set headers for SSE to work
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Connection", "keep-alive")

        # get uuid from url
        thread_id = self.get_argument("thread_id")
        model = self.get_argument("model")
        logging.info(f"Get chat request: thread_id={thread_id}, model={model}")

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
        else:
            del MESSAGE_STORAGE[thread_id]
        
        if real_model.startswith(("o1", "o3", "deepseek-r1")):
            if conversation[0]["role"] == "system":
                if conversation[0]["content"] != DEFAULT_SYSTEM_PROMPT:
                    conversation[1]["content"] = conversation[0]["content"]+"\n"+conversation[1]["content"]
                conversation = conversation[1:]

        billing_period = datetime.now().strftime("%Y-%m")
        user_cost = Global.audit_model.get_total_cost_by_user_id_billing_period(
            self.user_id, billing_period
        )
        user_limit = Global.audit_model.get_budget_by_user_id(self.user_id)

        if user_cost >= user_limit:
            del MESSAGE_STORAGE[thread_id]
            self.write(
                self.build_sse_message(
                    "You have exceeded your monthly budget.\nPlease contact to author.\nhttps://yishenggong.com/about-me"
                )
            )
            self.flush()
            return

        # create openai stream
        try:
            if model.lower().startswith("gpt"):
                stream = self.openai_text_stream(real_model, conversation)
            elif model.lower().startswith(("o1", "o3", "o4")):
                stream = self.openai_text_stream(real_model, conversation)
            elif model.lower().startswith("gemini"):
                stream = self.gemini_text_stream(real_model, conversation)
            elif model.lower().startswith("claude"):
                stream = self.claude_text_stream(real_model, conversation)
            else:
                self.build_return(400, {"error": "model is not supported"})
                return
        except Exception as e:
            logging.error(f"API error: {e}")
            self.write(
                self.build_sse_message(
                    "Sorry, I'm unable to get back to you this time due to service outage.\nCould you kindly contact to my author?\nhttps://yishenggong.com/about-me"
                )
            )
            self.flush()
            return

        assistant_message = ""
        input_token = 0
        output_token = 0
        async for resp in stream:
            if resp["type"] == "text" and resp["content"] is not None:
                text = resp["content"]
                assistant_message += text
                self.write(self.build_sse_message(text))
                self.flush()
            elif resp["type"] == "usage":
                input_token = resp["content"]["input_tokens"]
                output_token = resp["content"]["output_tokens"]
                logging.info(f"Get usage from API successfully")
        # save chat history
        conversation.append({"role": "assistant", "content": assistant_message})
        Global.chat_history_model.save_chat_history(
            self.user_id, thread_id, conversation
        )

        # audit cost
        if input_token == 0:
            for i in range(len(conversation) - 1):
                item = conversation[i]
                input_token += self.get_token_num(item["content"])

        if output_token == 0:
            output_token = self.get_token_num(assistant_message)
        logging.info(f"Input token: {input_token}, output_token: {output_token}")

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
        if not Global.invitation_code_model.validate_invitation_code(
            encrypted_invitation_code
        ):
            self.build_return(
                400, {"error": "Invitation code is not correct or expired"}
            )
            return
        else:
            user_id = Global.user_model.create_user()
            encrypted_pwd = encrypt_data(password, self.encrypt_key, self.encrypt_salt)
            Global.user_key_model.create_user(user_id, username, encrypted_pwd)
            Global.invitation_code_model.claim_invitation_code(
                encrypted_invitation_code
            )
            Global.audit_model.insert_budget_by_user_id(
                user_id, Global.config["default_monthly_budget"]
            )
            # init_default_prompt(user_id, Global.prompt_model)
            self.build_return(200)
            return


# handler for signin request (POST)
# POST: validate user, create session, set cookie
class SignInHandler(BaseHandler):
    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password = body.get("password")
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
# POST: insert invitation code to database
class InviteHandler(BaseHandler):
    def post(self):
        invitation_code = self.get_argument("invitation_code")
        if not invitation_code:
            self.build_return(
                400,
                {"error": "Invitation code is not provided"},
            )
            return
        else:
            encrypted_invitation_code = encrypt_data(
                invitation_code, self.encrypt_key, self.encrypt_salt
            )
            Global.invitation_code_model.insert_invitation_code(
                encrypted_invitation_code
            )
            self.build_return(200)
            return


# handler for forget password request (POST)
# POST: reset password using invitation code
class ForgetPasswordHandler(BaseHandler):
    def post(self):
        body = json.loads(self.request.body)
        username = body.get("username")
        password = body.get("password")
        invitation_code = body.get("invitation_code")
        if not username:
            self.build_return(400, {"error": "Username is not provided"})
            return
        elif not password:
            self.build_return(400, {"error": "Password is not provided"})
            return
        elif not invitation_code:
            self.build_return(400, {"error": "Invitation code is not provided"})
            return

        res = Global.user_key_model.validate_username(username)
        if not res:
            self.build_return(400, {"error": "Username is not found"})
            return

        # reset password
        encrypted_invitation_code = encrypt_data(
            invitation_code, self.encrypt_key, self.encrypt_salt
        )
        if not Global.invitation_code_model.validate_invitation_code(
            encrypted_invitation_code
        ):
            self.build_return(
                400, {"error": "Invitation code is not correct or expired"}
            )
            return

        Global.invitation_code_model.claim_invitation_code(encrypted_invitation_code)
        encrypted_pwd = encrypt_data(password, self.encrypt_key, self.encrypt_salt)
        Global.user_key_model.update_password_by_username(username, encrypted_pwd)
        self.build_return(200)
        return


# handler for audit request (GET)
# GET: get total cost and budget of current user from database
class AuditHandler(AuthHandler):
    def get(self):
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


# handler for random string request (GET)
# GET: generate random string
class RandomStringHandler(BaseHandler):
    def get(self):
        rand_string = "".join(random.choices(string.ascii_letters, k=32))
        self.build_return(200, rand_string)
        return


def make_app():
    return tornado.web.Application(
        [
            (r"/ping", PingHandler),
            (r"/auth", AuthHandler),
            (r"/file", FileHandler),
            (r"/chat", ChatHandler),
            (r"/prompt", PromptHandler),
            (r"/signin", SignInHandler),
            (r"/signup", SignUpHandler),
            (r"/invite", InviteHandler),
            (r"/forget_password", ForgetPasswordHandler),
            (r"/audit", AuditHandler),
            (r"/random", RandomStringHandler),
        ]
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", dest="config", default="backend-config.test.yaml")
    args = parser.parse_args()

    Global.config = read_config(args.config)
    log_file_path = Global.config["log_file_path"]
    setup_logger(log_file_path)
    logging.info(f"Read config successful: {Global.config}")

    APP_DATA_DB_CONN = sqlite3.connect(Global.config["app_data_db_name"])
    KEY_DB_CONN = sqlite3.connect(Global.config["key_db_name"])
    logging.info(
        f"Connected to database: [{Global.config['app_data_db_name']}] [{Global.config['key_db_name']}]"
    )

    Global.user_key_model = UserKeyModel(KEY_DB_CONN)
    Global.user_key_model.create_tables()
    Global.invitation_code_model = InvitationCodeModel(KEY_DB_CONN)
    Global.invitation_code_model.create_tables()
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

    init_price(APP_DATA_DB_CONN)

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
    logging.info(f"Starting Tornado server on 127.0.0.1:{Global.config['port']}")

    tornado.ioloop.IOLoop.current().start()
