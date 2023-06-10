import re 

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}")
def main():
    for i in range(100000):
        EMAIL_PATTERN.match("yisheng_gong@onmail.com")

if __name__ == "__main__":
    main()