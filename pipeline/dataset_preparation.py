
"""
dataset_preparation.py

This script provides a function to split a DataFrame containing text reviews and pseudo-labels
into training, validation, and test sets, and converts them into HuggingFace Datasets for use in
machine learning pipelines.
"""

from datasets import Dataset
from sklearn.model_selection import train_test_split

def prepare_dataset(df):
    """
    Splits a DataFrame into train, validation, and test sets and converts them to HuggingFace Datasets.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'review_text_clean' and 'pseudo_label'.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) as HuggingFace Datasets.
    """
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['review_text_clean'].tolist(),
        df['pseudo_label'].tolist(),
        test_size=0.4,
        random_state=42,
        stratify=df['pseudo_label']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    
    return train_dataset, val_dataset, test_dataset

train_ds, val_ds, test_ds = prepare_dataset(df)

print(f"Train size: {len(train_ds)}")
print(f"Validation size: {len(val_ds)}")
print(f"Test size: {len(test_ds)}")