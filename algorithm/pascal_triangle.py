def generate_1(numRows):
    # recursive solution
    if numRows == 0:
        return []
    if numRows == 1:
        return [[1]]
    if numRows == 2:
        return [[1], [1, 1]]
    res = generate_1(numRows - 1)
    last_row = res[-1]
    new_row = [1] + [last_row[i] + last_row[i + 1] for i in range(len(last_row) - 1)] + [1]
    res.append(new_row)
    return res

def generate_2(numRows):
    # iterative solution
    res = []
    if numRows == 0:
        return res
    first_row = [1]
    res.append(first_row)

    for i in range(1, numRows):
        last_row = res[i-1]
        curr_row = [1]
        for j in range(1, i):
            curr_row.append(last_row[j-1] + last_row[j])
        curr_row.append(1)
        res.append(curr_row)
    return res

def generate_3(numRows):
    # only give the last row
    if numRows == 0:
        return []
    if numRows == 1:
        return [1]
    if numRows == 2:
        return [1, 1]
    last_row = [1, 1]
    for i in range(2, numRows):
        new_row = [1] + [last_row[j] + last_row[j + 1] for j in range(len(last_row) - 1)] + [1]
        last_row = new_row
    return last_row

# todo: add lucas theorem
    

if __name__ == '__main__':
    print(generate_1(5))
    print(generate_2(5))
    print(generate_3(5))