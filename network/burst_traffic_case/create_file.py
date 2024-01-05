with open('small_file.txt', 'w') as f:
    f.write('a')

with open('big_file.txt', 'w') as f:
    # 1MB
    f.write('a'*1024*1024)