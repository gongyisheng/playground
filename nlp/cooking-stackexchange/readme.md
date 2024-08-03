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
`cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt`  

Before preprocessing:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.13733333333333334, 0.05939166786795445)
validate 2 result: (3000, 0.1085, 0.09384460141271443)
validate 3 result: (3000, 0.08511111111111111, 0.11042237278362405)
validate 4 result: (3000, 0.07183333333333333, 0.12426120801499208)
validate 5 result: (3000, 0.06606666666666666, 0.14285714285714285)
validate 6 result: (3000, 0.06316666666666666, 0.16390370477151506)
validate 7 result: (3000, 0.056523809523809525, 0.17111143145451924)
validate 8 result: (3000, 0.05325, 0.18422949401758684)
validate 9 result: (3000, 0.04981481481481481, 0.19388784777281245)
case1 result:  (('__label__baking', '__label__food-safety', '__label__bread', '__label__substitutions', '__label__equipment'), array([0.06983907, 0.06587588, 0.04016198, 0.03248651, 0.02998021]))
case2 result:  (('__label__baking', '__label__food-safety', '__label__bread', '__label__substitutions', '__label__equipment'), array([0.05927461, 0.04008852, 0.03691193, 0.02872915, 0.02530007]))
```

After preprocessing:
```
yisheng@deskmeetb600:~/playground/nlp/cooking-stackexchange$ python3 validate.py
validate 1 result: (3000, 0.16566666666666666, 0.07164480322906155)
validate 2 result: (3000, 0.125, 0.10811590024506271)
validate 3 result: (3000, 0.10077777777777777, 0.13074816202969583)
validate 4 result: (3000, 0.08441666666666667, 0.1460285425976647)
validate 5 result: (3000, 0.072, 0.1556868963528903)
validate 6 result: (3000, 0.06583333333333333, 0.17082312238719907)
validate 7 result: (3000, 0.059571428571428574, 0.1803373216087646)
validate 8 result: (3000, 0.05491666666666667, 0.1899956753639902)
validate 9 result: (3000, 0.050777777777777776, 0.19763586564797464)
case1 result:  (('__label__food-safety', '__label__baking', '__label__equipment', '__label__substitutions', '__label__chicken'), array([0.0986348 , 0.06364692, 0.03571884, 0.03141278, 0.02665346]))
case2 result:  (('__label__baking', '__label__bread', '__label__substitutions', '__label__food-safety', '__label__equipment'), array([0.09687459, 0.04882735, 0.03955015, 0.0356545 , 0.02949243]))
```