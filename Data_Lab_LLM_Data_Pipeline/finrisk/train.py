import torch
import os
from data import get_data
from model import get_model, get_loss
from utils import set_seed
from config import DATA_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, CHECKPOINT_PATH, SEED

def train():
    set_seed(SEED)

    # Lab 3 — DDP constants (simulated single process)
    rank = 0
    world_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Rank {rank} of {world_size} — device: {device}")

    # Data
    train_loader, val_loader, class_weights, le = get_data(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE
    )

    num_samples = len(train_loader.dataset)
    print(f"[train.py] Rank {rank} processing {num_samples} samples")  # Lab 3

    # Model
    model = get_model().to(device)
    criterion = get_loss(class_weights, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)  # Lab 3 — LR scaling

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if rank == 0:  # Lab 3 — only rank 0 logs and saves
            print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Lab 3 — checkpoint saved only on rank 0
    if rank == 0:
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"[train.py] Checkpoint saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()