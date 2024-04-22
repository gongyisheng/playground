from Crypto.Cipher import AES
import logging


def get_log_formatter():
    formatter = logging.Formatter(
        "%(levelname)s: [%(asctime)s][%(filename)s:%(lineno)s] %(message)s"
    )
    return formatter


def setup_logger(file_path: str = "backend-app.log"):
    logger = logging.getLogger()
    formatter = get_log_formatter()

    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logger.setLevel(logging.DEBUG)


# encrypt and decrypt utils functions
def encrypt_data(plaintext: str, key: str, salt: str) -> bytes:
    cipher = AES.new(key.encode("ascii"), AES.MODE_CFB, iv=salt.encode("ascii"))
    ciphertext = cipher.encrypt(plaintext.encode("ascii"))
    return ciphertext


def decrypt_data(ciphertext: bytes, key: str, salt: str) -> str:
    cipher = AES.new(key.encode("ascii"), AES.MODE_CFB, iv=salt.encode("ascii"))
    plaintext = cipher.decrypt(ciphertext).decode("ascii")
    return plaintext


if __name__ == "__main__":
    setup_logger()
    logging.info("test")
