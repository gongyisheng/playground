import fasttext

model = fasttext.load_model("cooking_model.bin")
print(model.test("cooking.valid", k=5))

print(model.predict("Why not put knives in the dishwasher?"))
print(model.predict("Which baking dish is best to bake a banana bread ?"))