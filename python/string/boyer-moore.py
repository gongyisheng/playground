# boyer-moore algorithm
def getBadMatch(p: str)-> dict:
    # get bad match table
    badMatch = {}
    length = len(p)
    for i in range(length):
        badMatch[p[i]] = max(1, length-i-1)
    return badMatch


def search(s: str, p: str)-> int:
    """
    s is the text
    p is the pattern
    if p is substring of s, return the index of first char
    """
    badMatch = getBadMatch(p)
    i = 0
    while i < len(s):
        if s[i] in badMatch.keys():
            j = 0
            while j<len(p):
                if s[i-j] != p[len(p)-j-1]:
                    i += badMatch[s[i]]
                    break
                else:
                    j += 1
            if j == len(p):
                return i-j+1
        else:
            i += len(p)
    return -1

if __name__ == '__main__':
    pattern = "9"
    text = "foofoobar"
    print(search(text, pattern))