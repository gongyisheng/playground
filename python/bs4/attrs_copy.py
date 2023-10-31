import copy
from bs4 import BeautifulSoup

def main():
    soup = BeautifulSoup('<b class="boldest">Extremely bold</b>', 'lxml')
    s1 = soup.prettify()
    original_attrs = []
    for tag in soup.find_all(True):
        original_attrs.append(copy.copy(tag.attrs))
        tag.attrs['test'] = 'test'
    i=0
    for tag in soup.find_all(True):
        tag.attrs = original_attrs[i]
        i = i + 1
    s2 = soup.prettify()
    print(s1==s2)

if __name__ == '__main__':
    main()