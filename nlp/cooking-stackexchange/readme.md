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
```python
import fasttext
model = fasttext.train_supervised(input="cooking.train")
model.save_model("cooking_model.bin")
```