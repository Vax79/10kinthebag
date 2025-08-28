import re
import pandas as pd
import os
import time
import sys

# --- Rule Definitions ---
ad_pattern = r"(https?://|www\.)|\b(discount|promo|visit)\b"
irrelevant_keywords = ["phone", "laptop", "movie", "unrelated"]
rant_phrases = ["never been", "haven't visited", "heard it's", "heard it is"]

# --- Detection Functions ---

def detect_advertisement(text: str) -> bool:
    """Detect if review contains ads or promotional content"""
    return bool(re.search(ad_pattern, text.lower()))

def detect_irrelevant(text: str) -> bool:
    """Detect if review contains unrelated keywords"""
    return any(word in text.lower() for word in irrelevant_keywords)

def detect_rant_without_visit(text: str) -> bool:
    """Detect if review implies ranting without visiting"""
    return any(phrase in text.lower() for phrase in rant_phrases)

def apply_policy_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all rules to a dataframe with a 'review_text' column.
    Returns dataframe with new columns: ad_flag, irrelevant_flag, rant_flag
    """
    df['ad_flag'] = df['text'].apply(detect_advertisement)
    df['irrelevant_flag'] = df['text'].apply(detect_irrelevant)
    df['rant_flag'] = df['text'].apply(detect_rant_without_visit)
    return df

#Removes rows that violates any policy rules and returns the filtered dataframe
#TEMP!! ALSO PRODUCES A DATAFRAME WITH THE FLAG COLUMNS FOR POLICY TESTING
def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df[~(df['ad_flag'] | df['irrelevant_flag'] | df['rant_flag'])].reset_index(drop=True)

    os.makedirs("data/filteredDataWithFlags", exist_ok=True)
    output_file = os.path.join("data/filteredDataWithFlags", f"cleaned_reviews_{int(time.time())}.csv")
    df.to_csv(output_file, index=False)

    return df_new.drop(['ad_flag', 'irrelevant_flag', 'rant_flag'], axis=1)

# --- Main method ---

def main(input_csv: str):
    
    df = pd.read_csv(input_csv)

    # Apply policy rules
    df = apply_policy_rules(df)

    # Filter dataset according to rules
    filtered_df = filter_dataset(df)

    os.makedirs("data/filteredData", exist_ok=True)
    output_file = os.path.join("data/filteredData", f"cleaned_reviews_{int(time.time())}.csv")
    filtered_df.to_csv(output_file, index=False)

    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Cleaned dataset saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python policy_module.py <input_csv>")
    else:
        main(sys.argv[1])
