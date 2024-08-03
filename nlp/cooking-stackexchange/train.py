import fasttext

model = fasttext.train_supervised(input="cooking.train", lr=0.5, epoch=50, wordNgrams=2, bucket=200000, dim=50, loss='ova')
model.save_model("cooking_model.bin")