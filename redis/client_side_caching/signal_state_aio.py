#! /usr/bin/env python
# coding=utf-8
import logging
import asyncio
import signal
import functools


ALIVE = True


def exit_handler(signum):
    logging.info("Exit signal got: %s, signum: %s", signal.getsignal(signum), signum)
    global ALIVE
    ALIVE = False


def register_exit_signal():
    exit_signals = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM]
    try:
        loop = asyncio.get_running_loop()
        for _signal in exit_signals:
            func = functools.partial(exit_handler, _signal)
            loop.add_signal_handler(_signal, func)
    except Exception as ex:
        logging.error("Errored in adding exit singal handlers, error[%s]", ex)
