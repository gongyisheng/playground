window_size = 2
data = []

for sentence in corpus:
    words = sentence.lower().split()
    for i, target in enumerate(words):
        context = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i and j >= 0 and j < len(words):
                context.append(words[j])
        if context:
            data.append((context, target))

print("Sample data:", data[:3])
