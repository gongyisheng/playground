import aiohttp
import asyncio
from contextvars import ContextVar
import os
import uuid
import logging

request = ContextVar("request")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s,%(msecs)03d][%(filename)s:%(lineno)d][%(process)s:%(threadName)s][%(request)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
    )

def request_tracer(results_collector):
    """
    Provides request tracing to aiohttp client sessions.
    :param results_collector: a dict to which the tracing results will be added.
    :return: an aiohttp.TraceConfig object.
    :example:
    >>> import asyncio
    >>> import aiohttp
    >>> from aiohttp_trace import request_tracer
    >>>
    >>>
    >>> async def func():
    >>>     trace = {}
    >>>     async with aiohttp.ClientSession(trace_configs=[request_tracer(trace)]) as client:
    >>>         async with client.get('https://github.com') as response:
    >>>             print(trace)
    >>>
    >>> asyncio.get_event_loop().run_until_complete(func())
    {'dns_lookup_and_dial': 43.3, 'connect': 334.29, 'transfer': 148.48, 'total': 526.08, 'is_redirect': False}
    """

    async def on_request_start(session, context, params):
        request.set(str(uuid.uuid4()).split('-')[0])
        context.on_request_start = session.loop.time()
        context.is_redirect = False

    async def on_connection_create_start(session, context, params):
        since_start = session.loop.time() - context.on_request_start
        context.on_connection_create_start = since_start

    async def on_request_redirect(session, context, params):
        since_start = session.loop.time() - context.on_request_start
        context.on_request_redirect = since_start
        context.is_redirect = True

    async def on_dns_resolvehost_start(session, context, params):
        since_start = session.loop.time() - context.on_request_start
        context.on_dns_resolvehost_start = since_start

    async def on_dns_resolvehost_end(session, context, params):
        since_start = session.loop.time() - context.on_request_start
        context.on_dns_resolvehost_end = since_start

    async def on_connection_create_end(session, context, params):
        since_start = session.loop.time() - context.on_request_start
        context.on_connection_create_end = since_start
    
    async def on_request_chunk_sent(session, context, params):
        since_start = session.loop.time() - context.on_request_start
        context.on_request_chunk_sent = since_start

    async def on_request_end(session, context, params):
        total = session.loop.time() - context.on_request_start
        context.on_request_end = total

        dns_lookup_and_dial = context.on_dns_resolvehost_end - context.on_dns_resolvehost_start
        connect = context.on_connection_create_end - dns_lookup_and_dial
        transfer = total - context.on_connection_create_end
        is_redirect = context.is_redirect

        results_collector['dns_lookup_and_dial'] = round(dns_lookup_and_dial * 1000, 2)
        results_collector['connect'] = round(connect * 1000, 2)
        results_collector['transfer'] = round(transfer * 1000, 2)
        results_collector['total'] = round(total * 1000, 2)
        results_collector['is_redirect'] = is_redirect

    trace_config = aiohttp.TraceConfig()

    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_redirect.append(on_request_redirect)
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)

    return trace_config

async def fetch(url):
    trace = {}
    async with aiohttp.ClientSession(trace_configs=[request_tracer(trace)]) as client:
        async with client.get(url) as response:
            print(trace)
            logging.info(f"URL: {url}", extra={"request": request.get()})

if __name__ == "__main__":
    urls = [
        'https://github.com',
        'https://www.google.com',
        'https://www.yahoo.com'
    ]
    setup_logging()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*[fetch(url) for url in urls]))