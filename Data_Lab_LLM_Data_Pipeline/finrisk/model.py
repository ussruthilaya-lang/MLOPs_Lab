import torch
from transformers import AutoModelForSequenceClassification

def get_model(num_labels=3, model_name="distilbert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model

def get_loss(class_weights, device):
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    return criterion