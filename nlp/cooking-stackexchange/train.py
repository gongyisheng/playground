import fasttext

model = fasttext.train_supervised(input="cooking.train")
model.save_model("cooking_model.bin")