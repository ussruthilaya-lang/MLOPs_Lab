import torch
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data
from model import get_model

def evaluate(checkpoint_path="checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, val_loader, class_weights, le = get_data(
        data_path="Sentences_AllAgree.txt",
        batch_size=16
    )

    # Load model from checkpoint
    model = get_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()