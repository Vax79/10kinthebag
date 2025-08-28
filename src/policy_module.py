import re
import pandas as pd
import os
import sys
import time 
from typing import Dict, List, Tuple, Set
from collections import Counter
import logging
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Rule Definitions ---
ad_patterns = [
    r"https?://[^\s]+",  
    r"www\.[^\s]+",      
    r"\b(visit|check out|go to)\s+[a-zA-Z0-9.-]+\.(com|net|org|co\.uk)\b",  
    r"\b(discount|promo|coupon|deal|offer|sale)\b",  
    r"\b(call|text|phone)\s*[:\-]?\s*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  
    r"\b(email|contact)\s*[:\-]?\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", 
    r"\$\d+\s*(off|discount)", 
    r"\b\d+%\s*(off|discount)\b",
    r"\b(free|buy|order|purchase|get yours|limited time)\b.*\b(now|today|call|visit)\b",
    r"\b(hire|job|employment|work opportunity)\b",
    r"dm me|message me|contact me",
    r"\b(affiliate|sponsored|partnership|referral)\b",
    r"\b(subscribe|follow me|like and share)\b",
    r"\b(my business|my company|my store|my website)\b"
]


irrelevant_keywords = {
    'technology': [
        "phone", "smartphone", "iphone", "android", "laptop", "computer", "tablet",
        "app", "software", "wifi", "bluetooth", "download",
        "ios", "windows", "mac", "browser", "chrome", "safari"
    ],
    'entertainment': [
        "movie", "film", "tv show", "series", "game", "gaming",
        "book", "novel", "music", "album", "song", "podcast", "streaming",
        "xbox", "playstation", "nintendo", "spotify"
    ],
    'fashion': [
        "clothes", "shirt", "shoes", "bag", "purse", "jewelry", "watch",
        "fashion", "outfit", "makeup", "cosmetics", "perfume", "brand"
    ],
    'off_topic': [
        "weather", "politics", "election", "celebrity", "sports team",
        "school", "homework", "exam", "job interview", "relationship", "dating",
        "health", "doctor", "medicine", "prescription", "personal life"
    ]
}

all_irrelevant_keywords = []
for category in irrelevant_keywords.values():
    all_irrelevant_keywords.extend(category)

rant_phrases = [
    "never been", "haven't been", "havent been", "have not been",
    "never visited", "haven't visited", "havent visited", "have not visited",
    "never went", "haven't went", "havent went", "have not went",
    " heard it's", "heard it is", "heard its", "heard they",
    "my friend said", "someone told me", "people say", "everyone says"
    "according to", "based on what i heard", "from what i hear",
    "supposedly", "apparently", "rumor has it"
]

fake_review_indicators = [
    r"^.{1,20}$",  # Very short reviews (less than 20 characters, likely generic)
]

copy_paste_indicators = [
    r"copy.*paste|paste.*copy",
    r"ctrl\s*[+]\s*[cv]|cmd\s*[+]\s*[cv]",  # Copy paste shortcuts
    r"(\b\w+\b.*){5,}\1",  # Repeated spam phrases/sentences 
]

profanity_list = [ # is this allowed idk delete if uw
    "wtf", "bs", "bullshit", "fuck", "shit", "ccb", "idiot"
]


# --- Detection Functions ---

def detect_advertisement(text: str) -> bool:
    """Detect if review contains ads or promotional content"""
    return bool(re.search(ad_patterns, text.lower())) 

def detect_irrelevant(text: str) -> bool:
    """Detect if review contains unrelated keywords"""
    text_lower = text.lower()
    
    # count irrelevant keywords
    irrelevant_count = sum(1 for word in all_irrelevant_keywords if word in text_lower)
    total_words = len(text_lower.split())
    
    # flag if multiple irrelevant keywords within a sentence
    if total_words > 0:
        irrelevant_ratio = irrelevant_count / total_words
        return irrelevant_ratio > 0.2 
    
    return irrelevant_count > 0
    return any(word in text.lower() for word in irrelevant_keywords)

def detect_rant_without_visit(text: str) -> bool:
    """Detect if review implies ranting without visiting"""
    return any(phrase in text.lower() for phrase in rant_phrases)

def detect_spam_content(text: str) -> bool:
    """Enhanced spam detection focusing on keyboard patterns and gibberish"""
    text_lower = text.lower()
    
    spam_patterns = [
        r'qwerty|asdfgh|zxcvbn',  # keyboard rows
        r'abcdef|123456|qazwsx',  # sequential patterns
        r'[bcdfghjklmnpqrstvwxyz]{6,}',  # long consonant sequences
        r'[aeiou]{5,}',  # excessive vowels
        r'(.)\1{4,}',  # same character repeated 5+ times
        r'^[^aeiou\s]{10,}$',  # words with no vowels (likely gibberish)
        r'^\d+$',  # only numbers
        r'^[^\w\s]+$',  # only special characters
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def detect_spam_repetition(text: str) -> bool:
    """Detect excessive word repetition"""
    
    # check for word repetitions
    word_counts = Counter(word for word in words if len(word) > 3)
    if word_counts:
        max_count = max(word_counts.values())
        if max_count > len(words) * 0.4:  
            return True
    
    # check for copy paste indicators
    text_lower = text.lower()
    for pattern in copy_paste_indicators:
        if re.search(pattern, text_lower):
            return True
    
    # check for repeated sentences/phrases
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 2:
        sentence_counts = Counter(s.strip().lower() for s in sentences if s.strip())
        if sentence_counts and max(sentence_counts.values()) > 1:
            return True
    
    return False

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
