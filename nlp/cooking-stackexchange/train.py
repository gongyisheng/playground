import fasttext

model = fasttext.train_supervised(input="cooking.train", epoch=25)
model.save_model("cooking_model.bin")