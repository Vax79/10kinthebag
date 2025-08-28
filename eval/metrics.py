import sys
import os

# Get absolute path to the src folder (sibling of eval)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)  # add src to Python path

# Now import the module
from policy_module import apply_policy_rules

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("eval/test_labels.csv")

df = apply_policy_rules(df)


# Example predictions (replace with actual outputs later)
# 0 = trustworthy, 1 = suspicious
# 'rule_predictions' are predictions outcome made by rules module
# 'nlp_predictions' are predictions outcome made by nlp model

df['rule_predictions'] = (df['ad_flag'] | df['irrelevant_flag'] | df['rant_flag']).astype(int)
# df['nlp_predictions'] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]   # temporary placeholder

# Calculate metrics for each method
# for method in ['rule_predictions', 'nlp_predictions']:
#     precision = precision_score(df['true_label'], df[method])
#     recall = recall_score(df['true_label'], df[method])
#     f1 = f1_score(df['true_label'], df[method])
#     print(f"{method} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


precision = precision_score(df['true_label'], df['rule_pred'])
recall = recall_score(df['true_label'], df['rule_pred'])
f1 = f1_score(df['true_label'], df['rule_pred'])

print("=== Rule-based Predictions Evaluation ===")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")