# boyer-moore algorithm
def getGoodSuffix(p: str)-> list:
    if len(p) == 0:
        return []
    if len(p) == 1:
        return [1]
    goodSuffix = [0 for _ in range(len(p))]
    i = len(p)-2
    j = len(p)-1
    while i >= 0:
        if p[i]!=p[j]:
            i -= 1
            j = len(p)-1
        else:
            if goodSuffix[j]==0:
                goodSuffix[j] = j-i
            i -= 1
            j -= 1
        print(f"i={i}, j={j}")
    print("good suffix table: ", goodSuffix)
    return goodSuffix

def getBadMatch(p: str)-> dict:
    # get bad match table
    badMatch = {}
    length = len(p)
    for i in range(length):
        badMatch[p[i]] = i
    print("bad match table: ", badMatch)
    return badMatch

def search(s: str, p: str)-> int:
    """
    s is the text
    p is the pattern
    if p is substring of s, return the index of first char
    """
    badMatch = getBadMatch(p)
    goodSuffix = getGoodSuffix(p)
    i = 0 # current guess
    while i <= len(s)-len(p):
        j = len(p)-1
        while j >= 0 and s[i+j] == p[j]:
            j -= 1
        if j < 0:
            return i
        else:
            i += max(j-badMatch.get(s[i+j], -1), goodSuffix[j])
    return -1

if __name__ == '__main__':
    pattern = "hell sell shell"
    text = "jasdahell sell shell"
    print(search(text, pattern))