import asyncio
import functools


class Example:
    def another_func(self):
        print("ANOTHER FUNC")

    def wrapper(func):
        @functools.wraps(func)
        async def wrap(self, *args, **kwargs):
            print("inside wrap")
            self.another_func()
            return await func(self, *args, **kwargs)

        return wrap

    @wrapper
    async def method(self):
        print("METHOD")


async def main():
    e = Example()
    await e.method()


if __name__ == "__main__":
    asyncio.run(main())
