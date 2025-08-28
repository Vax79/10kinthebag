
"""
dataset_preparation.py

This script provides a function to split a DataFrame containing text reviews and pseudo-labels
into training, validation, and test sets, and converts them into HuggingFace Datasets for use in
machine learning pipelines.
"""

from datasets import Dataset
from sklearn.model_selection import train_test_split

import pandas as pd

from dataset_preparation import prepare_dataset

# Load your DataFrame
df = pd.read_csv('data/cleanedData/reviews_cleaned.csv')

# Now call the function
train_dataset, val_dataset, test_dataset = prepare_dataset(df)

train_ds, val_ds, test_ds = prepare_dataset(df) # TODO: label clean data as df

print(f"Train size: {len(train_ds)}")
print(f"Validation size: {len(val_ds)}")
print(f"Test size: {len(test_ds)}")