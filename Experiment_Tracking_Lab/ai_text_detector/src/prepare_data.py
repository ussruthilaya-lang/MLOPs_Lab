"""
Data preparation — keeps raw text + label only.
TF-IDF in train.py handles all feature extraction.
Usage: python src/prepare_data.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'data_for_preprocessing.csv')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

def main():
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
    df.columns = ['text', 'label']
    df['label'] = df['label'].map({'AI': 1, 'Human': 0})
    df['text']  = df['text'].fillna('').str.strip()

    # Balance 2000 per class — keep full original texts
    balanced = pd.concat([
        df[df['label'] == 1].sample(2000, random_state=42),
        df[df['label'] == 0].sample(2000, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    balanced.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} | Shape: {balanced.shape}")
    print(f"Balance: {balanced['label'].value_counts().to_dict()}")
    print("Next: python src/train.py")

if __name__ == '__main__':
    main()