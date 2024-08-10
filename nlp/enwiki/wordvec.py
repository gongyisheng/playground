import fasttext
model = fasttext.load_model("enwiki_small.bin")

# It returns all words in the vocabulary, sorted by decreasing frequency.
w = model.words
print("Length of model words: ", len(w))
print("First 25 words: ", w[:25])

# Get the word vector
v = model.get_word_vector("meow")
print("Length of word vector: ", len(v))
print("Word vector: ", v)

# Get nearest neighbors
print("Nearest neighbors of orange: ", model.get_nearest_neighbors('orange', k=20))
print("Nearest neighbors of cat: ", model.get_nearest_neighbors('cat', k=20))

# Measure similarity