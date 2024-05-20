from tembo_pgmq_python import PGMQueue, Message


# please run in python > 3.9
def main():
    queue = PGMQueue(host="0.0.0.0", username="postgres", password="postgres")

    queue.create_queue("my_queue")
    msg_id = queue.send("my_queue", {"hello": "world"})
    read_message = queue.read("my_queue", vt=10)
    print(read_message.message)


if __name__ == "__main__":
    main()
