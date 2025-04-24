
# make a pandas datafrme
import pandas as pd

df = pd.DataFrame(columns=["label", "text"])

with open("smsspam.txt", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip().split("\t")
        label = line[0]
        text = line[1]
        if "," in text:
            text = "\"" + text + "\""
        if label == "ham":
            label = 0
        else:
            label = 1
        new_row = pd.DataFrame([[label, text]], columns=["label", "text"])
        df = pd.concat([df, new_row], ignore_index=True)

df.to_csv("./data/smsspam.csv", index=False)
        