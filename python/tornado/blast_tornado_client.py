import asyncio
import aiohttp
import json
import time
import string
import random

URL = "http://cluster-n0.local:8888/test"
CONCURRENCY = 100

class TestClient:
    def __init__(self):
        self.url = URL
        self.concurrent_requests = CONCURRENCY
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_large_payload(self, size_mb=1):
        """Generate a JSON payload of approximately the specified size in MB"""
        # Calculate approximate characters needed for 1MB (accounting for JSON overhead)
        chars_per_mb = 1024 * 1024
        target_chars = int(size_mb * chars_per_mb * 0.8)  # Account for JSON structure
        
        # Generate random data
        data = {
            "id": random.randint(1, 1000000),
            "timestamp": time.time(),
            "large_data": 'a'*target_chars,
            "metadata": {
                "client": "blast_tornado_client",
                "version": "1.0",
                "payload_size_mb": size_mb
            }
        }
        return data
    
    async def send_request(self, request_id):
        """Send a single POST request to the server"""
        payload = self.generate_large_payload()
        
        try:
            async with self.session.post(
                self.url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=1),
            ) as response:
                status = response.status
                result = await response.json()
                print(f"Request {request_id} returned status {status}")
                return status == 200
        except Exception as e:
            print(f"Request {request_id} failed: {e}")
            return False
    
    async def blast_requests(self, total_requests=None):
        """Send multiple concurrent requests to the server"""
        print(f"Starting {total_requests} requests with {self.concurrent_requests} concurrent connections")
        raw_total_requests = total_requests
        start_time = time.time()
        stop_flag = False

        while not stop_flag:
            tasks = [asyncio.create_task(self.send_request(i)) for i in range(self.concurrent_requests)]
            if total_requests is not None:
                total_requests -= self.concurrent_requests
                if total_requests <= 0:
                    stop_flag = True
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Execute all requests
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        print(f"\nResults:")
        print(f"Total requests: {raw_total_requests}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Requests per second: {total_requests/duration:.2f}")

async def main():
    # Configuration
    concurrent_requests = 100
    total_requests = None
    server_url = "http://localhost:8888"
    
    print(f"Tornado Client - Blasting {server_url}")
    print(f"Concurrent requests: {concurrent_requests}")
    print(f"Total requests: {total_requests}")
    print(f"Payload size: ~1MB per request")
    
    async with TestClient() as client:
        await client.blast_requests(total_requests)


if __name__ == "__main__":
    asyncio.run(main())