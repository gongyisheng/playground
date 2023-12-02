import logging

def get_log_formatter():
    formatter = logging.Formatter(
        "%(levelname)s: [%(asctime)s][%(module)s][%(filename)s:%(lineno)s]%(message)s"
    )
    return formatter

def setup_logger():

    logger = logging.getLogger()
    formatter = get_log_formatter()

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logger.setLevel(logging.DEBUG)