import json
import time
import random
import hashlib

def get_random_md5():
    return hashlib.md5(str(random.random()).encode('utf-8')).hexdigest()

test_flat_dict_100 = {get_random_md5(): get_random_md5() for i in range(100)}
test_flat_dict_1000 = {get_random_md5(): get_random_md5() for i in range(1000)}
test_flat_dict_10000 = {get_random_md5(): get_random_md5() for i in range(10000)}
test_flat_dict_100000 = {get_random_md5(): get_random_md5() for i in range(100000)}
test_flat_dict_1000000 = {get_random_md5(): get_random_md5() for i in range(1000000)}
test_flat_str_100 = json.dumps(test_flat_dict_100)
test_flat_str_1000 = json.dumps(test_flat_dict_1000)
test_flat_str_10000 = json.dumps(test_flat_dict_10000)
test_flat_str_100000 = json.dumps(test_flat_dict_100000)
test_flat_str_1000000 = json.dumps(test_flat_dict_1000000)

test_nest_dict_100 = {"a": test_flat_dict_100}

def get_nested_dict(depth, width):
    if depth == 1:
        return {str(i): i for i in range(width)}
    else:
        return {str(i): get_nested_dict(depth-1, width) for i in range(width)}

def test_dump(dict, round=10000):
    start_time = time.perf_counter()
    for i in range(round):
        json.dumps(dict)
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time)*1000
    print(f"once_time: {(duration_ms/round)}ms")
    print(f"total_time: {(duration_ms)}ms")

def test_load(str, round=10000):
    start_time = time.perf_counter()
    for i in range(round):
        json.loads(str)
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time)*1000
    print(f"once_time: {(duration_ms/round)}ms")
    print(f"total_time: {(duration_ms)}ms")

if __name__ == "__main__":
    # a = """{
    #     "a": "pig",
    #     "b": ["cat", "dog", {"c": "fish"}],
    #     "d": {"e": "bird", "f": "snake"}
    # }"""
    # print(json.loads(a))
    # test_dump(test_flat_dict_100)
    # test_dump(test_flat_dict_1000)
    # test_dump(test_flat_dict_10000, 1000)
    # test_dump(test_flat_dict_100000, 100)
    # test_dump(test_flat_dict_1000000, 10)
    test_load(test_flat_str_100)
    test_load(test_flat_str_1000)
    test_load(test_flat_str_10000, 1000)
    test_load(test_flat_str_100000, 100)
    test_load(test_flat_str_1000000, 10)
    import sys
    print("size of test_flat_str_100: ", sys.getsizeof(test_flat_str_100))
    print("size of test_flat_str_1000: ", sys.getsizeof(test_flat_str_1000))
    print("size of test_flat_str_10000: ", sys.getsizeof(test_flat_str_10000))
    print("size of test_flat_str_100000: ", sys.getsizeof(test_flat_str_100000))
    print("size of test_flat_str_1000000: ", sys.getsizeof(test_flat_str_1000000))

