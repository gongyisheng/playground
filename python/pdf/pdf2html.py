from pdf.pdf2html import pdf2html
import functools
from func_helper import timeout_run_func
import asyncio
from concurrent.futures import ProcessPoolExecutor


async def async_pdf2html(input_file, output_file):
    exe = ProcessPoolExecutor()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        exe, functools.partial(pdf2html, input_file, output_file)
    )
    exe.shutdown()


async def main(file_path):
    f = open(f"{file_path}.pdf", "rb")
    data = f.read()
    f.close()

    f = open(f"{file_path}-1.pdf", "wb")
    f.write(data)
    f.close()

    await timeout_run_func(
        60, async_pdf2html, f"{file_path}-1.pdf", f"{file_path}-1.html"
    )

    f = open(f"{file_path}-1.html", "rb")
    data = f.read()
    f.close()
    print("done")
    await asyncio.sleep(10)


if __name__ == "__main__":
    import sys

    asyncio.run(main(sys.argv[1]))
    # python pdf2html.py /Users/temp/Downloads/0694y00000LOxH8AAL-1 # 153k
    # python pdf2html.py /Users/temp/Downloads/i983-1_merged #7.5M
    # python pdf2html.py /Users/temp/Downloads/The-Effective-Engineer # 2.3M
    # python pdf2html.py /Users/temp/Downloads/postfix-the-definitive-guide # 9.6M
