import fasttext
model = fasttext.train_unsupervised('/home/yisheng/data/enwiki-clean')
model.save_model("enwiki.bin")