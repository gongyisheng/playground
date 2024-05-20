import json

s = "{1}"


def main():
    try:
        json.loads(s)
    except Exception as e:
        print("error: %s" % e)


main()
