
import aiohttp

SERPER_API_KEY=""

async def search_google(query):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = { "q": query }

    # to asynchronously fetch data
    async with aiohttp.ClientSession() as session:
        async with session.post("https://google.serper.dev/search", headers=headers, json=data) as response:
            if response.status == 200:
                data = await response.json()
                results = []
                for item in data.get("organic", [])[:5]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", "")
                    })
                return results
            else:
                return {"error": f"HTTP error {response.status}"}

if __name__ == "__main__":
    import asyncio
    query = "What is the capital of France?"
    result = asyncio.run(search_google(query))
    print(result)