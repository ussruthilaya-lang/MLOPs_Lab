import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class FinancialDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

def get_data(data_path, tokenizer_name="distilbert-base-uncased", batch_size=16):
    # Lab 1 — Simulate streaming with shuffle buffer instead of full random load
    df = pd.read_csv(data_path, sep="@", encoding="latin-1", header=None, names=["sentence", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle buffer simulation

    # Lab 2 — First pass: count labels, compute class weights before training
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["label_id"]),
        y=df["label_id"]
    )
    print(f"[data.py] Label distribution estimated. Class weights: {class_weights.round(2)}")

    # Lab 1 — Manual 80/20 sequential split (streaming doesn't support stratified splits)
    split = int(0.8 * len(df))
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]
    print(f"[data.py] Train size: {len(train_df)}, Val size: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = FinancialDataset(train_df["sentence"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_dataset = FinancialDataset(val_df["sentence"].tolist(), val_df["label_id"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, class_weights, le