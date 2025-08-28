import re
import pandas as pd

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?\,\-\:]', '', text)
    text = text.strip()
    return text

# Example usage:
# df = pd.read_csv('reviews_dataset.csv')
df['review_text_clean'] = df['review_text'].apply(clean_text)
df = df[df['review_text_clean'].str.len() > 0]
print(f"Dataset after cleaning: {len(df)} rows")
