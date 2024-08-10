import fasttext
model = fasttext.load_model("enwiki_small.bin")

# It returns all words in the vocabulary, sorted by decreasing frequency.
def inspect_words():
    w = model.words
    print("Length of model words: ", len(w))
    print("First 25 words: ", w[:25])

# Get the word vector
def inspect_wordvec(word):
    v = model.get_word_vector(word)
    print("Length of word vector: ", len(v))
    print(f"Word vector of {word}: ", v)

# Get nearest neighbors
def inspect_nn(words):
    for word in words:
        print(f"Nearest neighbors of {word}: ", model.get_nearest_neighbors(word, k=20))

# Measure similarity
import numpy as np
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def inspect_similarity(word_pairs): 
    for pair in word_pairs:
        w1, w2 = pair
        v1 = model.get_word_vector(w1)
        v2 = model.get_word_vector(w2)
        print(f"Similarity of ({w1}, {w2}) is, ", cosine_similarity(v1, v2))

if __name__ == "__main__":
    # inspect_words()
    # inspect_wordvec("cat")
    # inspect_nn(["orange", "cat"])
    inspect_similarity([
        ("orange", "cat"),
        ("fat", "cat"),
        ("piggy", "cat"),
        ("naughty", "cat")
    ])