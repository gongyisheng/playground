def main():
    try:
        raise Exception("Exception from try")
    except Exception as e:
        raise e
    finally:
        print("finally")
        # return 3 # if uncommented, 3 will be returned


if __name__ == "__main__":
    main()
