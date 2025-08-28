import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("eval/test_labels.csv")
df2 = pd.read_csv("data/filteredDataWithFlags/cleaned_reviews_1756378203.csv")
sample_df = df2.sample(n=30, random_state=42)
print(sample_df)
# Example predictions (replace with actual outputs later)
# 0 = trustworthy, 1 = suspicious
# 'rule_predictions' are predictions outcome made by rules module
# 'nlp_predictions' are predictions outcome made by nlp model

sample_df['rule_predictions'] = (sample_df['ad_flag'] | sample_df['irrelevant_flag'] | sample_df['rant_flag']).astype(int)
# df['nlp_predictions'] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]   # temporary placeholder

# Calculate metrics for each method
# for method in ['rule_predictions', 'nlp_predictions']:
#     precision = precision_score(df['true_label'], df[method])
#     recall = recall_score(df['true_label'], df[method])
#     f1 = f1_score(df['true_label'], df[method])
#     print(f"{method} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


precision = precision_score(df['true_label'], sample_df['rule_predictions'])
recall = recall_score(df['true_label'], sample_df['rule_predictions'])
f1 = f1_score(df['true_label'], sample_df['rule_predictions'])

print("=== Rule-based Predictions Evaluation ===")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")