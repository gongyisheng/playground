try:
    # generator exit inherits from base exception
    # if except Exception is used, it still can't catch GeneratorExit
    # if except BaseException or nothing is used, it can catch GeneratorExit
    raise GeneratorExit()
except Exception as e:
    print("exception")
    pass

# result:
# Traceback (most recent call last):
#   File "try_catch_no_exception.py", line 5, in <module>
#     raise GeneratorExit()
# GeneratorExit
