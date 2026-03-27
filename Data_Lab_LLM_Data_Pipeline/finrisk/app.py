import streamlit as st
import torch
from transformers import AutoTokenizer
from model import get_model

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer, device

LABELS = {0: "🔴 Negative", 1: "⚪ Neutral", 2: "🟢 Positive"}

EXAMPLES = [
    "The company reported record profits this quarter, exceeding analyst expectations.",
    "The firm filed for bankruptcy following a series of failed investments.",
    "The board meeting concluded with no changes to the current strategy.",
    "Operating profit rose to EUR 13.1 mn from EUR 8.0 mn in the previous year.",
    "The company faces significant legal risks due to ongoing litigation.",
]

# Sidebar — project info
st.sidebar.title("About this Project")
st.sidebar.markdown("""
**Distributed Financial Risk Classifier**

Built as part of MLOps Labs 1–4.

**Model:** DistilBERT fine-tuned on Financial PhraseBank

**Training highlights:**
- 2264 financial sentences
- Class imbalance: 61% neutral / 25% positive / 13% negative
- Weighted CrossEntropyLoss to correct gradient bias
- Shuffle buffer + sequential 80/20 split (streaming-style)

**Validation Results:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.86 | 0.98 | 0.92 |
| Neutral | 0.99 | 0.97 | 0.98 |
| Positive | 0.93 | 0.91 | 0.92 |
| **Macro avg** | **0.93** | **0.95** | **0.94** |

**Engineering:**
- Modular scripts: data.py, model.py, train.py, evaluate.py, config.py
- DDP-ready training with rank-0 checkpoint saving
- Reproducible with set_seed(42)
""")

# Main
st.title("📊 Financial Risk Classifier")
st.markdown("Classify financial sentences as **Positive**, **Neutral**, or **Negative** using a fine-tuned DistilBERT model.")

st.markdown("---")

# Example sentences
st.subheader("Try an example")
example = st.selectbox("Select an example sentence", ["-- Select --"] + EXAMPLES)

st.subheader("Or enter your own")
user_input = st.text_area("Input sentence", value=example if example != "-- Select --" else "", height=100)

if st.button("Classify", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        model, tokenizer, device = load_model()

        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()
            pred = torch.argmax(probs).item()

        st.markdown("---")
        st.subheader(f"Prediction: {LABELS[pred]}")

        st.write("**Confidence scores:**")
        for i, label in LABELS.items():
            st.progress(float(probs[i]), text=f"{label}: {probs[i]:.2%}")