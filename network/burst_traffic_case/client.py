# Before running this script, please make sure that redis server is running.
import redis


def main():
    set_client = redis.Redis(host="172.31.92.214", port=6379, db=0)
    get_client = redis.Redis(host="172.31.92.214", port=6379, db=0)
    for i in range(10):
        set_client.set(f"small_key_{i}", "a" * i * 100)  # 100B, 200B, 300B, ..., 1KB
    set_client.set("big_key", "a" * 1024 * 1024 * 10)  # 10MB

    for i in range(10):
        resp = get_client.get(f"small_key_{i}")
        print(f"get small_key complete. length={len(resp)}")

    for i in range(2):
        resp = get_client.get("big_key")
        print(f"get big_key complete. length={len(resp)}")


if __name__ == "__main__":
    main()
