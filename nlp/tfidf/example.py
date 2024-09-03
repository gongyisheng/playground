from nltk.tokenize import word_tokenize
import numpy as np

sample_text = [
    'Topic sentences are similar to mini thesis statements. Like a thesis statement', 
    'a topic sentence has a specific main point. Whereas the thesis is the main point of the essay, the topic sentence is the main point of the paragraph. Like the thesis statement, a topic sentence has a unifying function. But a thesis statement or topic sentence alone doesn\'t guarantee unity.', 
    'An essay is unified if all the paragraphs relate to the thesis,\'whereas a paragraph is unified if all the sentences relate to the topic sentence.'
]

sentences = []
word_set = []

for sent in sample_text:
    words = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
    sentences.append(words)
    for word in words:
        if word not in word_set:
            word_set.append(word)
# Set of words
word_set = set(word_set)
# total documents in our corpus
total_docs = len(sample_text)
print('Total documents: ', total_docs)
print('Total words: ', len(word_set))

word_index = {}
word_array = []
for i, word in enumerate(word_set):
    word_index[word] = i
    word_array.append(word)

def count_dict(sentences):
    count_dict = {}
    for word in word_set:
        count_dict[word] = 0
    for sent in sentences:
        for word in sent:
            count_dict[word] += 1
    return count_dict

word_count = count_dict(sentences)
print(word_count)

def term_frequency(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance / N

def inverse_document_frequency(word):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_docs / word_occurance)

def tf_idf(sentence):
    vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = term_frequency(sentence, word)
        idf = inverse_document_frequency(word)
        vec[word_index[word]] = tf * idf
    return vec

vectors = []
for sent in sentences:
    vectors.append(tf_idf(sent))

print(vectors[0], len(vectors[0]))
score = {word_array[i]:vectors[0][i] for i in range(len(vectors[0])) if vectors[0][i] != 0}
print(sentences[0], score)