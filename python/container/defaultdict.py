"""Demo script for defaultdict(dict)."""

from collections import defaultdict

# ---- 1. Basic usage ----
print("1. Basic: defaultdict(dict)")
d = defaultdict(dict)

# No KeyError — auto-creates empty dict for missing keys
d["user1"]["name"] = "Alice"
d["user1"]["age"] = 30
d["user2"]["name"] = "Bob"

print(d)
# {'user1': {'name': 'Alice', 'age': 30}, 'user2': {'name': 'Bob'}}

# ---- 2. Compare with normal dict ----
print("\n2. Normal dict fails:")
normal = {}
try:
    normal["user1"]["name"] = "Alice"
except KeyError as e:
    print(f"  KeyError: {e}")

# workaround with setdefault
normal.setdefault("user1", {})["name"] = "Alice"
print(f"  After setdefault: {normal}")

# ---- 3. Nested grouping example ----
print("\n3. Group students by class and subject:")
scores = [
    ("classA", "math", 90),
    ("classA", "math", 85),
    ("classA", "english", 78),
    ("classB", "math", 92),
    ("classB", "english", 88),
]

grouped = defaultdict(dict)
for cls, subject, score in scores:
    grouped[cls].setdefault(subject, []).append(score)

for cls, subjects in grouped.items():
    print(f"  {cls}: {dict(subjects)}")

# ---- 4. Other default factories ----
print("\n4. Other factories:")

dd_list = defaultdict(list)
dd_list["fruits"].append("apple")
dd_list["fruits"].append("banana")
print(f"  defaultdict(list):   {dict(dd_list)}")

dd_int = defaultdict(int)
dd_int["count"] += 1
dd_int["count"] += 1
print(f"  defaultdict(int):    {dict(dd_int)}")

dd_set = defaultdict(set)
dd_set["tags"].add("python")
dd_set["tags"].add("python")  # duplicate ignored
dd_set["tags"].add("ml")
print(f"  defaultdict(set):    {dict(dd_set)}")

dd_lambda = defaultdict(lambda: "N/A")
dd_lambda["exists"] = "hello"
print(f"  defaultdict(lambda): exists={dd_lambda['exists']}, missing={dd_lambda['missing']}")

# ---- 5. Nested depth limits ----
print("\n5. defaultdict(dict) is only 1 level deep:")
a = defaultdict(dict)
a["1"]["2"] = "ok"
print(f"  a['1']['2'] = {a['1']['2']}")
try:
    a["x"]["y"]["z"] = "fail"
except KeyError as e:
    print(f"  a['x']['y']['z'] → KeyError: {e}")

# ---- 6. Unlimited depth with recursive defaultdict ----
print("\n6. Unlimited depth:")

def deep_dict():
    return defaultdict(deep_dict)

d = deep_dict()
d["a"]["b"]["c"]["d"]["e"] = "5 levels deep!"
print(f"  d['a']['b']['c']['d']['e'] = {d['a']['b']['c']['d']['e']}")

d["config"]["db"]["postgres"]["host"] = "localhost"
d["config"]["db"]["postgres"]["port"] = 5432
d["config"]["db"]["redis"]["host"] = "localhost"
print(f"  d['config']['db']['postgres'] = {dict(d['config']['db']['postgres'])}")
print(f"  d['config']['db']['redis']    = {dict(d['config']['db']['redis'])}")
