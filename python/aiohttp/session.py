import aiohttp
import asyncio


async def test_multiple_urls(urls):
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for url in urls:
            try:
                response = await session.request("get", url)
                content = await response.content.read()
                print(f"Process web url success. url[{url}. content[{content[:128]}]")
            except Exception as e:
                print(f"Process web url error. url[{url}]")

async def test_connector(url):
    # Create new session with connection pool limits
    # Note: TCPConnector is a connection pool
    #   it will reuse connections if force_close=False
    #   and may cause load unbalance issue
    # To test it, you need to run on tornado server
    # simple server like http.server close connections after response complete by default
    timeout = aiohttp.ClientTimeout(
        connect=5,
        sock_connect=60
    )
    connector = aiohttp.TCPConnector(
        ssl=False,
        limit=100,
        force_close=True
    )
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    for i in range(10):
        async with session.get(url) as response:
            dict_result = await response.text()
        await asyncio.sleep(1)
    
    await session.close()

if __name__ == "__main__":
    # urls = ["www.google.com", "www.abc.com"]
    # asyncio.run(test_multiple_urls(urls))

    url = "http://localhost:8888"
    asyncio.run(test_connector(url))
