def binary_search(sorted_array, target):
    left = 0
    right = len(sorted_array) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_array[mid] == target:
            return mid
        elif sorted_array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    # return left-1 # return lower bound 
    # return left # return upper bound 
    return -1 # return -1 if not exist

def sqrt(x):
    left = 0
    right = x
    while left <= right:
        mid = (left + right) // 2
        bid = mid * mid
        if bid == x:
            return bid
        elif bid < x:
            left = mid + 1
        elif bid > x:
            right = mid - 1
    return left-1


if __name__ == "__main__":
    sorted_array = [1,3,5,6]
    target = 4
    print(binary_search(sorted_array, target))
    print(sqrt(12))