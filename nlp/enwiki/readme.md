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
**A nice feature is that you can also query for words that did not appear in your data**
**Words are represented by sum of substrings**
```
yisheng@pc:~/playground/nlp/enwiki$ python3 wordvec.py
Similarity of (environment, envirobmental) is,  0.78040737
```

## Skipgram vs cbow
### skipgram: we try to predict the context words using the main word
eg.  
- sentence:   
    the pink horse is eating  
- word pair:  
    (the, pink), (the, horse),  
	(pink, the), (pink, horse), (pink, is),  
    (horse, the), (horse, pink), (horse, is), (horse, eating),  
    (is, pink), (is, horse), (is, eating),  
    (eating, horse), (eating, is)  

model structure:    
1. V=number of words in vocabulary  
2. E=desired size of word embeddings  

- input: one-hot encoding word vector (1xV)  
- weight vector1: transform word vector into hidden layer (VxE)  
- hidden layer: new word embedding (1xE)  
- weight vector2: transform word embedding into output layer (ExV)  
- output layer: context words in one-hot encoding (several 1xV)  

**The higher this size is, the more information the embeddings will capture, but the harder it will be to learn it**

### cbow(continuous-bag-of-words): From the context words, we want our model to predict the main word
model structure:
1. V=number of words in vocabulary  
2. E=desired size of word embeddings  

- input: context words in one-hot encoding word vector (several 1xV)
- weight vector1: transform word vector into hidden layer (VxE)
- hidden layer: new word embedding (1xE)
- weight vector2: transform word embedding into output layer (ExV)
- output layer: main word in one-hot encoding (1xV)

```
# train cbow model
import fasttext
model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean', "cbow")
```

### comparison
- Skip-Gram works well with small datasets, and can better represent less frequent words
- CBOW is found to train faster than Skip-Gram, and can better represent more frequent words
- In practice, we observe that skipgram models works better with subword information than cbow (usually rare words are important)

ref: https://www.baeldung.com/cs/word-embeddings-cbow-vs-skip-gram

## Playing with the parameters
- minn/maxn: substring size contained in a word, 3-6 is popular (eg, beautiful, beautifully may share same substring)
- dim: size of the word embedding, 100-300 is popular
- epoch: time to loop the dataset
- lr: learning rate, [0.01, 1]
- thread: core number to use
```
import fasttext
model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean', minn=3, maxn=6, dim=300, lr=0.05, epoch=10)
```


