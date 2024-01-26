import logging

def get_log_formatter():
    formatter = logging.Formatter(
        "%(levelname)s: [%(asctime)s][%(filename)s:%(lineno)s] %(message)s"
    )
    return formatter

def setup_logger():

    logger = logging.getLogger()
    formatter = get_log_formatter()

    fh = logging.FileHandler("backend-app.log")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    setup_logger()
    logging.info("test")