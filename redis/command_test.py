import redis


# hkeys
def test_hkeys():
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.hset("test", "test_field1", "1")
    r.hset("test", "test_field2", "2")
    val = r.hkeys("test")
    print(val)

# exists
def test_exists():
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.hset("test", "test_field1", "1")
    val = bool(r.exists("test"))
    print(val)
    val = bool(r.exists("test1"))
    print(val)



if __name__ == "__main__":
    #test_hkeys()
    test_exists()
