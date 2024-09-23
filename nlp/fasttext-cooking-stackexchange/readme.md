# Text classifier for cooking.stackexchange.com using FastText
## Pre-requisites
- Build fasttext python module
```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip install .
```
- Download dataset
```
wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz
head cooking.stackexchange.txt
```
- Split dataset into train and test
```
wc cooking.stackexchange.txt
15404  169582 1401900 cooking.stackexchange.txt

head -n 12404 cooking.stackexchange.txt > cooking.train
tail -n 3000 cooking.stackexchange.txt > cooking.valid
```
## Train model
`python3 train.py`
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   89106 lr:  0.000000 avg.loss: 10.047649 ETA:   0h 0m 0s
```

## Validate model
`python3 validate.py`
```
(3000, 0.13833333333333334, 0.0598241314689347)
The first number: precision
The second number: recall
```

## Text preprocessing
```
cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
head -n 12404 cooking.preprocessed.txt > cooking.train
tail -n 3000 cooking.preprocessed.txt > cooking.valid
```

Before preprocessing validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.13833333333333334, 0.0598241314689347)
validate 2 result: (3000, 0.10666666666666667, 0.09225890154245352)
validate 3 result: (3000, 0.08666666666666667, 0.11244053625486522)
validate 4 result: (3000, 0.07183333333333333, 0.12426120801499208)
validate 5 result: (3000, 0.06606666666666666, 0.14285714285714285)
validate 6 result: (3000, 0.063, 0.16347124117053483)
validate 7 result: (3000, 0.05657142857142857, 0.17125558598817933)
validate 8 result: (3000, 0.05354166666666667, 0.18523857575320743)
validate 9 result: (3000, 0.04981481481481481, 0.19388784777281245)
case1 result:  (('__label__baking', '__label__food-safety', '__label__bread', '__label__substitutions', '__label__equipment'), array([0.06803396, 0.06552208, 0.03904324, 0.03344319, 0.03047537]))
case2 result:  (('__label__baking', '__label__food-safety', '__label__bread', '__label__substitutions', '__label__equipment'), array([0.05889857, 0.04035925, 0.03631106, 0.02937603, 0.02584211]))
```

After preprocessing validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.164, 0.07092403056076113)
validate 2 result: (3000, 0.12483333333333334, 0.10797174571140263)
validate 3 result: (3000, 0.10044444444444445, 0.13031569842871557)
validate 4 result: (3000, 0.08433333333333333, 0.1458843880640046)
validate 5 result: (3000, 0.07213333333333333, 0.15597520542021046)
validate 6 result: (3000, 0.06588888888888889, 0.17096727692085917)
validate 7 result: (3000, 0.05952380952380952, 0.18019316707510452)
validate 8 result: (3000, 0.05504166666666667, 0.19042813896497046)
validate 9 result: (3000, 0.05085185185185185, 0.1979241747152948)
case1 result:  (('__label__food-safety', '__label__baking', '__label__equipment', '__label__substitutions', '__label__chicken'), array([0.09797386, 0.06312738, 0.03583983, 0.0310504 , 0.02633232]))
case2 result:  (('__label__baking', '__label__bread', '__label__substitutions', '__label__food-safety', '__label__equipment'), array([0.09635726, 0.04812203, 0.03949984, 0.03542671, 0.02944902]))
```

## Add epoch
By default, fastText sees each training example only five times during training, which is pretty small, given that our training set only have 12k training examples. The number of times each examples is seen (also known as the number of epochs) can be increased to make the model better
```
model = fasttext.train_supervised(input="cooking.train", epoch=25)
```

Train output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   93429 lr:  0.000000 avg.loss:  7.235476 ETA:   0h 0m 0s
```

Validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.5186666666666667, 0.22430445437509008)
validate 2 result: (3000, 0.3715, 0.3213204555283264)
validate 3 result: (3000, 0.29155555555555557, 0.37826149632405937)
validate 4 result: (3000, 0.23825, 0.412137811734179)
validate 5 result: (3000, 0.20373333333333332, 0.4405362548652155)
validate 6 result: (3000, 0.1771111111111111, 0.45956465330834656)
validate 7 result: (3000, 0.1575238095238095, 0.47686319734755656)
validate 8 result: (3000, 0.14158333333333334, 0.4898371053769641)
validate 9 result: (3000, 0.1292222222222222, 0.5029551679400317)
case1 result:  (('__label__equipment', '__label__knives', '__label__food-safety', '__label__cookware', '__label__food-identification'), array([0.57279921, 0.03021949, 0.02339702, 0.02196073, 0.01164233]))
case2 result:  (('__label__baking', '__label__equipment', '__label__bread', '__label__pie', '__label__crust'), array([0.75866228, 0.02363421, 0.01694055, 0.01510086, 0.01197328]))
```

## Add learning rate
Another way to change the learning speed of our model is to increase (or decrease) the learning rate of the algorithm. This corresponds to how much the model changes after processing each example. A learning rate of 0 would mean that the model does not change at all, and thus, does not learn anything. Good values of the learning rate are in the range 0.1 - 1.0
```
model = fasttext.train_supervised(input="cooking.train", lr=1.0)
```

Train output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   89121 lr:  0.000000 avg.loss:  6.291508 ETA:   0h 0m 0s
```

Validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.5736666666666667, 0.2480899524290039)
validate 2 result: (3000, 0.43116666666666664, 0.3729277785786363)
validate 3 result: (3000, 0.3402222222222222, 0.441401182067176)
validate 4 result: (3000, 0.28158333333333335, 0.4870981692374225)
validate 5 result: (3000, 0.24013333333333334, 0.5192446302436211)
validate 6 result: (3000, 0.2086111111111111, 0.5413002738936139)
validate 7 result: (3000, 0.18466666666666667, 0.5590312815338042)
validate 8 result: (3000, 0.16633333333333333, 0.5754648983710537)
validate 9 result: (3000, 0.15107407407407408, 0.588006342799481)
case1 result:  (('__label__equipment', '__label__knives', '__label__sharpening', '__label__cleaning', '__label__knife-skills'), array([0.41527042, 0.29635984, 0.04575914, 0.02549994, 0.0151588 ]))
case2 result:  (('__label__baking', '__label__bread', '__label__cupcakes', '__label__cake', '__label__crust'), array([0.87620056, 0.01242683, 0.00884113, 0.00807289, 0.0077443 ]))
```

## Try both
```
model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25)
```

Train output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   93340 lr:  0.000000 avg.loss:  4.253905 ETA:   0h 0m 0s
```

Validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.589, 0.25472106097736774)
validate 2 result: (3000, 0.4678333333333333, 0.4046417759838547)
validate 3 result: (3000, 0.3751111111111111, 0.4866657056364423)
validate 4 result: (3000, 0.3105, 0.5371197924174715)
validate 5 result: (3000, 0.26493333333333335, 0.5728701167651723)
validate 6 result: (3000, 0.23, 0.5967997693527461)
validate 7 result: (3000, 0.20495238095238094, 0.6204411128729999)
validate 8 result: (3000, 0.18408333333333332, 0.6368747297102494)
validate 9 result: (3000, 0.1674074074074074, 0.6515784921435779)
case1 result:  (('__label__knives', '__label__food-safety', '__label__storage-method', '__label__equipment', '__label__oil'), array([0.41230521, 0.30281836, 0.05949405, 0.05396916, 0.04856721]))
case2 result:  (('__label__bananas', '__label__baking', '__label__equipment', '__label__rising', '__label__bread'), array([9.26999331e-01, 6.79376051e-02, 8.10887199e-04, 6.39245030e-04, 3.81109014e-04]))
```

## word n-grams
Finally, we can improve the performance of a model by using word bigrams, instead of just unigrams. This is especially important for classification problems where word order is important, such as sentiment analysis. For example, in the sentence, 'Last donut of the night', the unigrams are 'last', 'donut', 'of', 'the' and 'night'. The bigrams are: 'Last donut', 'donut of', 'of the' and 'the night'.
```
model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2)
```

Train output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   90487 lr:  0.000000 avg.loss:  3.187120 ETA:   0h 0m 0s
```

Validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.6016666666666667, 0.26019893325645094)
validate 2 result: (3000, 0.4695, 0.40608332132045555)
validate 3 result: (3000, 0.3698888888888889, 0.4798904425544183)
validate 4 result: (3000, 0.304, 0.525875738791985)
validate 5 result: (3000, 0.2594, 0.5609052904713854)
validate 6 result: (3000, 0.22505555555555556, 0.5839700158569987)
validate 7 result: (3000, 0.19904761904761906, 0.6025659506991495)
validate 8 result: (3000, 0.179125, 0.6197203402046995)
validate 9 result: (3000, 0.16233333333333333, 0.6318293210321465)
case1 result:  (('__label__knives', '__label__equipment', '__label__storage-method', '__label__cleaning', '__label__food-safety'), array([0.21641926, 0.17434357, 0.09015955, 0.05382202, 0.04091829]))
case2 result:  (('__label__bananas', '__label__equipment', '__label__baking', '__label__muffins', '__label__grilling'), array([0.74399638, 0.07120986, 0.03634853, 0.02284147, 0.008861  ]))
```

## Scaling things up
A potential solution to make the training faster is to use the hierarchical softmax, instead of the regular softmax. The hierarchical softmax is a loss function that approximates the softmax with a much faster computation.
```
model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
```

Train output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread: 1928044 lr:  0.000000 avg.loss:  2.253972 ETA:   0h 0m 0s
```

Validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.5873333333333334, 0.25400028830906735)
validate 2 result: (3000, 0.44383333333333336, 0.38388352313680263)
validate 3 result: (3000, 0.3451111111111111, 0.4477439815482197)
validate 4 result: (3000, 0.28229704950825135, 0.4882514055067032)
validate 5 result: (3000, 0.24214738246082027, 0.5234251117197636)
validate 6 result: (3000, 0.20983879933296276, 0.5441833645668156)
validate 7 result: (3000, 0.18613295210864902, 0.5630676084762866)
validate 8 result: (3000, 0.16752856785386605, 0.5790687617125558)
validate 9 result: (3000, 0.1521489227574443, 0.591466051607323)
case1 result:  (('__label__knives', '__label__equipment', '__label__cleaning', '__label__pressure-cooker', '__label__organization'), array([0.42612314, 0.38081208, 0.06341884, 0.02273137, 0.0154162 ]))
case2 result:  (('__label__bananas', '__label__baking', '__label__equipment', '__label__recipe-scaling', '__label__seasoning'), array([0.59435242, 0.17348757, 0.05823667, 0.04332617, 0.02526515]))
```

## Multi-label classification
When we want to assign a document to multiple labels, we can still use the softmax loss and play with the parameters for prediction, namely the number of labels to predict and the threshold for the predicted probability. However playing with these arguments can be tricky and unintuitive since the probabilities must sum to 1.  

A convenient way to handle multiple labels is to use independent binary classifiers for each label. This can be done with -loss one-vs-all or -loss ova.  
```
# Note: use lr=0.5 to avoid NaN
model = fasttext.train_supervised(input="cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
```

Train output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 train.py
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:  160873 lr:  0.000000 avg.loss:  4.184714 ETA:   0h 0m 0s
```

Validate output:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.606, 0.262072942194032)
validate 2 result: (3000, 0.46, 0.3978665129018308)
validate 3 result: (3000, 0.35844444444444445, 0.4650425255874297)
validate 4 result: (3000, 0.2886666666666667, 0.4993513045985296)
validate 5 result: (3000, 0.24346666666666666, 0.5264523569266254)
validate 6 result: (3000, 0.2095, 0.5436067464321753)
validate 7 result: (3000, 0.1828095238095238, 0.553409254721061)
validate 8 result: (3000, 0.162125, 0.5609052904713854)
validate 9 result: (3000, 0.14629629629629629, 0.5694104079573302)
case1 result:  (('__label__knives', '__label__bread', '__label__equipment', '__label__food-safety', '__label__hot-chocolate'), array([6.95517510e-02, 1.00000034e-05, 1.00000034e-05, 1.00000034e-05, 1.00000034e-05]))
case2 result:  (('__label__baking', '__label__bananas', '__label__equipment', '__label__convection', '__label__gluten-free'), array([1.00001001, 0.99409896, 0.51562995, 0.08510906, 0.02676929]))
```