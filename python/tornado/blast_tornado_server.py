import asyncio
import json
import hashlib
import time
import tornado.web

class TestHandler(tornado.web.RequestHandler):
    async def post(self):
        # try:
        #     json_data = json.loads(self.request.body)
        # except json.JSONDecodeError:
        #     json_data = {}
        
        # # Proof of work - hash generation
        # self._hash_generation(json_data)
        
        # self.set_status(200)
        # self.write({"status": "success"})
        self.write("Hello, world")
    
    def _hash_generation(self, json_data):
        # Generate multiple hashes as proof of work
        dummy_data = json_data
        data = f"pow_work_{time.time()}"
        for i in range(100000):
            data = hashlib.sha256(data.encode()).hexdigest()
        dummy_data = json_data.get("variable", "")
        return dummy_data

def make_app():
    return tornado.web.Application([
        (r"/test", TestHandler),
    ])


async def main():    
    app = make_app()
    app.listen(8888)
    print("Server running on http://localhost:8888")
    
    # Keep the server running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())