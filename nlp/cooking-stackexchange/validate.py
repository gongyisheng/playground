import fasttext

model = fasttext.load_model("cooking_model.bin")
model.test("cooking.valid", k=5)