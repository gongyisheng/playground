import asyncio
import time
import aiomysql

async def connect():
    connect_timeout = 10
    autocommit = True
    db = {
        'host': '404notfound.net',
        'port': 3306,
        'username': 'test',
        'password': 'test',
        'dbname': 'example',
        'charset': 'utf8mb4',
    }
    db_conn = await aiomysql.connect(
        host=db['host'],
        user=db['username'],
        password=db['password'],
        db=db['dbname'],
        charset=db['charset'],
        port=db['port'],
        connect_timeout=connect_timeout,
        autocommit=autocommit,
    )
    return db_conn

async def tick():
    last_tick = int(time.time()*1000)
    while True:
        now_tick = int(time.time()*1000)
        print('tick, interval:', now_tick - last_tick)
        last_tick = now_tick
        await asyncio.sleep(1)


async def main():
    task = asyncio.create_task(tick())
    try:
        db_conn = await connect()
    except Exception as e:
        print('connect error:', e)
    await task

if __name__ == '__main__':
    asyncio.run(main())