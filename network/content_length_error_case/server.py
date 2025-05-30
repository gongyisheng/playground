import asyncio
import logging
import random

logging.basicConfig(level=logging.DEBUG)


async def fugazi_web(rd, wr):
    alive_choices = (False, True, True)
    chunked_choices = (False, True)

    while True:
        req = await rd.read(8192)

        keepalive = random.choice(alive_choices)
        chunked = random.choice(chunked_choices)

        if chunked:
            hdrs = (
                'Transfer-encoding: chunked',
            )
        else:
            content_len = random.randint(8192 + 5, 16385 + 5)
            hdrs = (
                f'Content-Length: {content_len}',
            )

        resp = '\r\n'.join((
            'HTTP/1.1 200 OK',
            f'Connection: {keepalive and "keep-alive" or "close"}',
            *hdrs,
            '',
            '',
        )).encode()

        if not chunked:
            # tag body onto response
            resp += b'A' * content_len

        wr.write(resp)
        await wr.drain()

        if chunked:
            while random.random() < 0.95:
                chunk_len = random.randint(1, 8192 + 5)
                chunk = f'{chunk_len:x}\r\n{"A" * chunk_len}\r\n'.encode()
                wr.write(chunk)
                await wr.drain()

            wr.write(b'0\r\n\r\n')
            await wr.drain()

        if not keepalive:
            break

    # print('closing')
    wr.close()
    await wr.wait_closed()


async def main():
    srv = await asyncio.start_server(
        fugazi_web,
        '127.0.0.1',
        12345,
    )

    async with srv:
        await srv.serve_forever()

if __name__ == '__main__':
    asyncio.run(main())