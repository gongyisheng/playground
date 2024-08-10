import fasttext
model = fasttext.load_model("enwiki_small.bin")

# It returns all words in the vocabulary, sorted by decreasing frequency.
print("Length of model words: ", len(model.words))
print("First 25 words: ", model.words[:25])

# Get the word vector
print(model.get_word_vector("meow"))