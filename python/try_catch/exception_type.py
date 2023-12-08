a = [1,2,3]
try:
    a[4]
except Exception as e:
    print(str(e.__class__))