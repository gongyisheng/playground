import asyncio
import yappi
# yappi supports for asyncio better

async def foo():
    await asyncio.sleep(1.0)
    await baz()
    await asyncio.sleep(0.5)

async def bar():
    await asyncio.sleep(2.0)

async def baz():
    await asyncio.sleep(1.0)

async def main():
    await asyncio.gather(foo(), bar(), baz())

if __name__ == "__main__":
    yappi.set_clock_type("wall")
    with yappi.run():
        #asyncio.run(foo())
        #asyncio.run(bar())
        asyncio.run(main())
    stat = yappi.get_func_stats()
    stat.save("output.prof", type="pstat")
    # visualize : pip install snakeviz, snakeviz result.out