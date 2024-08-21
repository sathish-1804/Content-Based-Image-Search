import torch
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define function to generate embeddings
def generate_embeddings(texts):
    # Tokenize input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Forward pass through the model to obtain embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings from the last layer
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    return embeddings

# Attempt to read the descriptions from the CSV file with different encodings
try:
    df = pd.read_csv("descriptions.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("descriptions.csv", encoding='latin1')

# Generate embeddings for descriptions
embeddings = generate_embeddings(df["Description"].tolist())

# Save embeddings to a pickle file
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
