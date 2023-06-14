import aiohttp
import asyncio

urls = ['https://abc.top', "https://www.google.com"]
#urls = ['https://www.google.com']
urls = urls*3
async def main():
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for url in urls:
            try:
                response = await session.request('get', url)
                content = await response.content.read()
                print(f'Process web url success. url[{url}. content[{content[:128]}]')
            except Exception as e:
                print(f'Process web url error. url[{url}]')
                # raise e

if __name__ == '__main__':
    asyncio.run(main())