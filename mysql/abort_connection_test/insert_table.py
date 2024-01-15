import asyncio
import aiomysql

db_host = "127.0.0.1"
db_port = 3306
db_user = ""
db_password = ""
db_name = "test"

payload = "a" * 32


async def main():
    # connect to mysql
    conn = await aiomysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        db=db_name,
        autocommit=True,
    )
    # create cursor
    async with conn.cursor() as cursor:
        for i in range(400000):
            print(f"Inserting row {i}")
            # execute sql
            await cursor.execute(
                f'INSERT INTO fetchtable_test (payload) VALUES ("{payload}")'
            )
    # close connection
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
