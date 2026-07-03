import pandas as pd

df = pd.read_csv("data/math500_result.csv")
df = df[['answer', 'naive_answer', 'power_answer', 'power_acceptance_ratio']]

df[df['answer']==df['naive_answer']]

df[df['answer']==df['power_answer']]
