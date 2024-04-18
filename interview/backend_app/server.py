# server for customized chatbot
from datetime import datetime
import json
import logging
import sqlite3
import uuid

import yaml
import tornado.ioloop
import tornado.web

from utils import setup_logger, encrypt_data, decrypt_data


# read yaml config file
def read_config(file_path):
    with open(file_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        logging.debug("Read config successful. env: %s" % data["environment"])
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

# base handler for all requests, cors configs
class BaseHandler(tornado.web.RequestHandler):

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


# handler for chat request (POST and GET)
# POST: save prompt and context to memory storage, validation
# GET:  send prompt to openai and transfer completion SSE stream to client, audit, save chat history
class RecordingHandler(BaseHandler):
    def post(self):
        pass

    def get(self):
        # get retrive code from request
        retrive_code = self.get_argument("retrive_code", None)
        self.finish()


def make_app():
    return tornado.web.Application(
        [
            (r"/", RecordingHandler),
        ]
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    setup_logger()

    parser = ArgumentParser()
    parser.add_argument("--config", dest="config", default="backend-config.test.yaml")
    args = parser.parse_args()

    Global.config = read_config(args.config)

    APP_DATA_DB_CONN = sqlite3.connect(Global.config["app_data_db_name"])
    KEY_DB_CONN = sqlite3.connect(Global.config["key_db_name"])
    logging.info(
        "Connected to database: [%s] [%s]"
        % (Global.config["app_data_db_name"], Global.config["key_db_name"])
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
    logging.info(f"Starting Tornado server on 127.0.0.1:{Global.config['port']}")

    tornado.ioloop.IOLoop.current().start()
