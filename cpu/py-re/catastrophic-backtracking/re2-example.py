import re2 as re

EMAIL_PATTERN = re.compile(r"^([a-zA-Z0-9]+\s?)*$")
def main():
    for i in range(1000):
        EMAIL_PATTERN.match("tOSMD SDAASDA SDSKD=DSAD ASDJ=ASDJLAS")

if __name__ == "__main__":
    main()