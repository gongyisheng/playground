import asyncio
import aiomysql

db_host = "127.0.0.1"
db_port = 3306
db_user = ""
db_password = ""
db_name = "test"


async def fetch_table(sql: str):
    # connect to mysql
    conn = await aiomysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        db=db_name,
        connect_timeout=10,
        autocommit=True,
    )
    # create cursor
    async with conn.cursor(aiomysql.SSDictCursor) as cursor:
        # execute sql
        await cursor.execute(sql)
        rows = await cursor.fetchmany(size=8)
        while rows:
            for row in rows:
                yield row
            rows = await cursor.fetchmany(size=8)
    # close connection
    conn.close()


async def main():
    sql = "SELECT * FROM fetchtable_test where id > 0"
    async for row in fetch_table(sql):
        print(row)
        if row["id"] == 1:
            await asyncio.sleep(120)


if __name__ == "__main__":
    asyncio.run(main())
