import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("eval/test_labels.csv")

# Example predictions (replace with actual outputs later)
# 0 = trustworthy, 1 = suspicious
# 'rule_predictions' are predictions outcome made by rules module
# 'nlp_predictions' are predictions outcome made by nlp model
df['rule_predictions'] = [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]  # temporary placeholder
df['nlp_predictions'] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]   # temporary placeholder

# Calculate metrics for each method
for method in ['rule_predictions', 'nlp_predictions']:
    precision = precision_score(df['true_label'], df[method])
    recall = recall_score(df['true_label'], df[method])
    f1 = f1_score(df['true_label'], df[method])
    print(f"{method} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
