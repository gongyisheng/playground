import fasttext

def train_large():
    model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean')
    model.save_model("enwiki.bin")

def train_small():
    model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-small-clean')
    model.save_model("enwiki_small.bin")

train_small()