import fasttext

model = fasttext.load_model("cooking_model.bin")
model.test("cooking.valid", k=5)

model.predict("Why not put knives in the dishwasher?")
model.predict("Which baking dish is best to bake a banana bread ?")