import pickle
import json
import time


def main(round):
    f = open("/Users/temp/Downloads/log4j-active (2).txt", "rb")
    content = f.read()

    _tmp1 = None
    _tmp2 = None

    start = time.time()
    for i in range(round):
        _tmp1 = pickle.dumps(content)
    end = time.time()
    print(f"[pickle dump] cost {end-start} s")

    start = time.time()
    for i in range(round):
        _tmp2 = pickle.loads(_tmp1)
    end = time.time()
    print(f"[pickle load]cost {end-start} s")
    print(_tmp2 == content)

    start = time.time()
    for i in range(round):
        _tmp1 = json.dumps(content.decode("utf-8"))
    end = time.time()
    print(f"[json dump] cost {end-start} s")

    start = time.time()
    for i in range(round):
        _tmp2 = json.loads(_tmp1).encode("utf-8")
    end = time.time()
    print(f"[json load]cost {end-start} s")
    print(_tmp2 == content)


if __name__ == "__main__":
    main(100)
