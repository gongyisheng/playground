import logging

init_count = 0
add_count = 0


class PageQueue:
    def __init__(self):
        global init_count
        init_count += 1

    async def lpush(self):
        global add_count
        add_count += 1


def summary_stat():
    logging.info(f"init_count={init_count}, add_count={add_count}")
