import json
import requests

headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://www.sec.gov",
    "Priority": "u=1, i",
    "Referer": "https://www.sec.gov/",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

# def cik_lookup(ticker: str):
#     url = "https://www.sec.gov/cgi-bin/cik_lookup"
#     response = requests.post(url, headers=headers, data={"company": ticker})
#     print(response.text)


def format_cik(cik: str):
    return f"{cik:0>10}"


def cik_lookup(ticker: str):
    url = "https://efts.sec.gov/LATEST/search-index"
    data = json.dumps({"keysTyped": ticker.upper(), "narrow": "true"})
    response = requests.post(url, headers=headers, data=data)
    result = response.json()
    if result["timed_out"] or result["_shards"]["successful"] == 0:
        return None
    resp_tickers = set(result["hits"]["hits"][0]["_source"]["tickers"].split(", "))
    if ticker not in resp_tickers:
        return None
    cik = result["hits"]["hits"][0]["_id"]
    return format_cik(cik)


def company_facts(cik: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    f = open("company_facts.json", "w", encoding="utf_8")
    f.write(json.dumps(response.json()))
    f.close()
    return response.json()


if __name__ == "__main__":
    cik = cik_lookup("AAPL")
    print(company_facts(cik))
