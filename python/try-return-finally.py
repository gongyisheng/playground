def main(doRaise=False):
    try:
        print("try")
        if doRaise:
            raise Exception("Exception from try")
        return 1
    except Exception as e:
        print("except")
        return 2
    finally:
        print("finally")
        # return 3 # if uncommented, 3 will be returned

if __name__ == "__main__":
    print(main(doRaise=False))
    print(main(doRaise=True))
