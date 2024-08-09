# Word representations project
## Download data
The data source is Wikipedia's articles
```
mkdir ~/data
cd ~/data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 &
bzip2 -d enwiki-latest-pages-articles.xml.bz2 &    // Note: this will not preserve original archive file
mv ~/data/enwiki-latest-pages-articles.xml ~/data/enwiki
```
## Pre-process
Process the html / xml raw text
```
perl wikifil.pl ~/data/enwiki > ~/data/enwiki-clean
```

Inspect the data
```
yisheng@deskmeetb600:~/data$ head -c 80 ~/data/enwiki-clean
anarchism is a political philosophy and movement that is against all forms of a
```

## Train word vectors
```
import fasttext
model = fasttext.train_unsupervised('~/data/enwiki-clean')
model.save_model("enwiki.bin")
```

Inspect words and word vector
```
import fasttext
model = fasttext.load_model("enwiki.bin")

# It returns all words in the vocabulary, sorted by decreasing frequency.
print(model.words)

# Get the word vector
print(model.get_word_vector("the"))
```

