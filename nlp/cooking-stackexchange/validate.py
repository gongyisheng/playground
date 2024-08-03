import fasttext

model = fasttext.load_model("cooking_model.bin")
result = model.test("cooking.valid", k=5)
print("validate result: ", result)

result = model.predict("Why not put knives in the dishwasher?", k=5)
print("case1 result: ", result)
result = model.predict("Which baking dish is best to bake a banana bread?", k=5)
print("case2 result: ", result)