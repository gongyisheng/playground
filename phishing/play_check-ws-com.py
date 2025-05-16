import asyncio
import websockets
import json
import time
import random
import logging

def gen_random_hk_phone_number():
    return f"{random.choice(['8', '5', '4'])}{random.randint(1000000, 9999999)}"

def gen_random_6_numbers():
    return "".join([str(random.randint(1, 9)) for _ in range(6)])

async def send_one_request():
    url = "wss://check-ws.com/socket/4SF4ASV46DS1V6489VA4DS6V"
    phone_number = gen_random_hk_phone_number()

    origin = "https://check-ws.com"
    user_agent_header = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"

    try:
        async with websockets.connect(url, origin=origin, user_agent_header=user_agent_header) as websocket:
            payload = {
                "type": "phone-code",
                "uuid": f"IP{int(time.time()*1000)}-{gen_random_6_numbers()}",
                "phone": f"+852{phone_number}",
                "country": "852",
                "phoneNumber": phone_number,
                "host": "check-ws.com"
            }

            # print("Phone number:", phone_number)

            await websocket.send(json.dumps(payload))
            # print("Data sent. Waiting for responses...")

            # Receive messages until connection is closed
            async for message in websocket:
                logging.info(f"Received:{message}")
                break

    except websockets.exceptions.ConnectionClosed as e:
        logging.error(f"Connection closed:{e}")
    except Exception as e:
        logging.error(f"Error:{e}")

async def send_one_request_with_timeout():
    try:
        await asyncio.wait_for(send_one_request(), timeout=15)
    except asyncio.TimeoutError:
        logging.error("Timeout occurred")

async def main(concurrency=10):
    tasks = [asyncio.create_task(send_one_request_with_timeout()) for _ in range(concurrency)]
    await asyncio.gather(*tasks)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Run it
asyncio.run(main())
