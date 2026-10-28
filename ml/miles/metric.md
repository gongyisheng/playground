# Miles Metrics

## repetition_frac
miles defines “repetitive” through compression
```
raw = window.encode("utf-8")
ratio = len(raw) / len(zlib.compress(raw, level=9))
is_repetitive = ratio > 1
```

for long rollouts:
```
Window 1:       0–10,000
Window 2:   5,000–15,000
Window 3:  10,000–20,000
...
Window 19: 90,000–100,000

For each window, it calculates original_bytes / compressed_bytes.

- If any window has a ratio > 10, the entire response contributes 1.
- If no window exceeds 10, the response contributes 0.
```