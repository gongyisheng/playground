# Text classifier for cooking.stackexchange.com using FastText
## Pre-requisites
- Build fasttext python module
```bash
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


## Text preprocessing
```
cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
head -n 12404 cooking.preprocessed.txt > cooking.train
tail -n 3000 cooking.preprocessed.txt > cooking.valid
```


Before preprocessing:
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

After preprocessing:
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