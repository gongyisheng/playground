import aiohttp
import asyncio

urls = ['https://united.com/en/us/checkin/e/e05a48d3a0760eb55072178129f90f7d/ebp/2/38313fbd94228e906495c6dbc263350b2cd8110b07f6419447c7cb55a8264ca1', "https://www.google.com"]
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