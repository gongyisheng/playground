# Before running this script, please make sure that server is running.
# python -m http.server 8000

import requests

small_file_url = 'http://172.1.1.1/small_file.txt'
big_file_url = 'http://172.1.1.1/big_file.txt'

def main():
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
    session.mount('http://', adapter)
    
    for _ in range(10):
        response = session.get(small_file_url)
        print(f"Get small file response. code={response.status_code}, length={len(response.content)}")
    
    for _ in range(1):
        response = session.get(big_file_url)
        print(f"Get big file response. code={response.status_code}, length={len(response.content)}")