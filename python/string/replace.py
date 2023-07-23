import time

s = "abcaskjhknjaccoiekasldkasn,calsjdjnjkcejncalksjdefhkjahcbamsfjkajlwonascbjahjsdljkasfkl;k;l\ndsakjdgkbdjkhkakhhcalslklkajklsd\tdlksajnclkajlksdjkl"

def str_replace1(s):
    s = s.replace('\n', ' ')
    s = s.replace('\t', ' ')
    s = s.replace(',', ' ')
    return s

def str_replace2(s):
    s = s.replace('\n', ' ').replace('\t', ' ').replace(',', ' ')
    return s

def main(round):
    start = time.time()
    for i in range(round):
        _s = s
        _s = str_replace1(_s)
    end = time.time()
    print("str_replace1: ", end - start)

    start = time.time()
    for i in range(round):
        _s = s
        _s = str_replace2(_s)
    end = time.time()
    print("str_replace2: ", end - start)

if __name__ == "__main__":
    main(1000000)