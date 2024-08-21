import streamlit as st
import pickle
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch

# Set environment variable to address the OpenMP warning
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Convert Tensor object to NumPy array
embeddings = embeddings.numpy()

embedding_size = embeddings.shape[1]
n_clusters = 3  # Decrease the number of clusters to address the warning
num_results = 5

# Create FAISS index
quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(
    quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT,
)
index.train(embeddings)
index.add(embeddings)

# Define the search function
def search(query):
    query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**query_tokens)
    query_embedding = output.last_hidden_state.mean(dim=1).numpy()
    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, num_results)
    return indices[0]

# Streamlit UI
st.title("Image Search")
query = st.text_input("Enter your search query:")
if st.button("Search"):
    indices = search(query)
    for i, index in enumerate(indices):
        st.image(f"images/{index}.jpg", caption=f"Result {i+1}")
