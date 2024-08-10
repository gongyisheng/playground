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
model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean')
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

Inspect Output:
```
yisheng@pc:~/playground/nlp/enwiki$ python3 wordvec.py
['the', 'of', 'one', 'zero', 'and', 'in', 'two', 'a', 'nine', 'to', 'is', 'eight', 'three', 'four', 'five', 'six', 'seven', 'for', 'are', 'as', 'was', 's', 'with', 'by', 'from']
[-0.07939239  0.09653837  0.22372544  0.16011845 -0.1287736  -0.08914521
 -0.45821008 -0.2527027  -0.1559536  -0.26215678 -0.09395579  0.06869064
 -0.17072806 -0.06316316 -0.3726137  -0.0051136  -0.13838841  0.03436428
  0.1061983  -0.12363543  0.12507957  0.08602201  0.30533016  0.01326842
 -0.13504317  0.02098932  0.20269214 -0.36423954 -0.07256772  0.30299184
  0.21057566 -0.05874498 -0.09813377 -0.20411912  0.10108342  0.2990998
 -0.0853412  -0.01064696 -0.07070319  0.09914397 -0.10967136  0.31570482
  0.1031241   0.04691444 -0.08933676 -0.5185983   0.2296672  -0.0018015
 -0.36503765 -0.04936389  0.04346406  0.07122956  0.1472588   0.05205015
  0.16742358 -0.0882655   0.17377302 -0.11354169  0.09396518 -0.12195109
  0.0843197   0.4470378   0.1354175   0.30283815  0.05910857  0.11689328
 -0.06196691  0.12597375 -0.11636457  0.0734896   0.15679339  0.13017073
  0.03316979 -0.3901237  -0.22026618  0.09468966  0.13410209 -0.00409082
  0.21215588  0.0593515   0.08999079 -0.27802116 -0.25878018  0.03348326
 -0.07304306 -0.0200186   0.05441337 -0.3500039  -0.13057813  0.2976526
  0.2871939   0.30215597 -0.0070382   0.02623713 -0.43378437 -0.40853783
 -0.14576787 -0.22968893  0.25703436 -0.25776097]
```

## Skipgram vs cbow
- skipgram: predict a target word thanks to a nearby word (using a random close-by word, can be before or after)
- cbow(continuous-bag-of-words): cbow model predicts the target word according to its context (using a surrounding window)
```
# train cbow model
import fasttext
model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean', "cbow")
```

