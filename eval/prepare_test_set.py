import pandas as pd

df = pd.read_csv("data/cleanedData/reviews_cleaned.csv")

sample_df= df.sample(n=30, random_state=42)
test_set = sample_df[['text']].copy()
test_set['true_label'] = 0
test_set.to_csv("eval/test_labels.csv", index=False)
