import fasttext

def train_large():
    model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean')
    model.save_model("enwiki.bin")

def train_small():
    model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-small-clean')
    model.save_model("enwiki_small.bin")

def train_with_param():
    model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-small-clean', minn=3, maxn=6, dim=300, lr=0.05, epoch=10)
    model.save_model("enwiki_small_optimized.bin")

# train_small()
train_with_param()