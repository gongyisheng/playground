def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    piviot = arr[0]
    left = []
    right = []
    for num in arr[1:]:
        if num < piviot:
            left.append(num)
        else:
            right.append(num)
    return quick_sort(left) + [piviot] + quick_sort(right)

def merge_sort(arr):
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    if len(arr) <= 1:
        return arr
    
    # Split the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    return merge(left_half, right_half)


if __name__ == '__main__':
    arr = [5, 3, 7, 2, 8, 4, 1, 6]
    print(quick_sort(arr))
    print(merge_sort(arr))