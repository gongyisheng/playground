import fasttext
model = fasttext.load_model("enwiki_small.bin")

# It returns all words in the vocabulary, sorted by decreasing frequency.
print(model.words[:25])

# Get the word vector
print(model.get_word_vector("the"))