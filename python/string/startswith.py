import time

text = "aaabbbcccdddeee"
prefix1 = ("aaa", "bbb", "ccc", "ddd", "eee")
prefix2 = ("bbb", "ccc", "ddd", "eee", "aaa")

def match1(prefix):
    for p in prefix:
        if text.startswith(p):
            return True
    return False

def match2(prefix):
    return text.startswith(prefix)

def main(prefix):
    start = time.time()
    for i in range(1000000):
        match1(prefix)
    end = time.time()
    print("match1: ", end - start)

    start = time.time()
    for i in range(1000000):
        match2(prefix)
    end = time.time()
    print("match2: ", end - start)

if __name__ == "__main__":
    main(prefix1)
    main(prefix2)

# Result: 
# temp@Orange-cats-shared-mac string % python startswith.py
# match1:  0.14869117736816406
# match2:  0.12369990348815918
# match1:  0.4314742088317871
# match2:  0.14927101135253906