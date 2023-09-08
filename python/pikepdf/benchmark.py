import asyncio
import time
from pathlib import Path
import pikepdf
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    # filename=logfile,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%H:%M:%S'
)


def benchmark_pikepdf():
    root_directory = Path("/Users/temp/Downloads/test-pdf")
    cost_ms = []
    for p in root_directory.glob('*.pdf'):
        try:
            absolute_path = p.resolve()
            stime = time.time()
            pdf = pikepdf.open(absolute_path)
            logging.info(f'allow.extract={pdf.allow.extract}, {pdf.is_encrypted}')
            pdf.save(f"{p.name}.crack")
            cost_ms.append(int((time.time() - stime) * 1000))
        except Exception as e:
            logging.error(f'{e}')
    cost_ms.sort()
    idx90th = int(len(cost_ms) * 0.90)
    idx95th = int(len(cost_ms) * 0.95)
    idx99th = int(len(cost_ms) * 0.99)
    logging.info(f'test_file_count={len(cost_ms)}, 90th={cost_ms[idx90th]}ms, 95th={cost_ms[idx95th]}ms, 99th={cost_ms[idx99th]}ms')


async def main():
    benchmark_pikepdf()


if __name__ == '__main__':
    asyncio.run(main())