import fasttext
model = fasttext.train_unsupervised('~/data/enwiki-clean')
model.save_model("enwiki.bin")