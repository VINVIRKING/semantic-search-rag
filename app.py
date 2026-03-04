from transformers import pipeline
import streamlit as st
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Title
st.title("🚀 Semantic Search Engine (FAISS + Embeddings)")

# Load model and dataset
@st.cache_resource
def load_model_and_data():

    # Load dataset
    dataset = load_dataset("ag_news", split="train[:1000]")
    documents = dataset["text"]

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Text generator (LLM)
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    return model, index, documents, generator


# Load everything
model, index, documents, generator = load_model_and_data()

# User input
query = st.text_input("Enter your search query:")

# Slider
top_k = st.slider("Number of results:", 1, 5, 3)

# Search button
if st.button("Search") and query:

    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    st.subheader("Results:")

    for i in range(top_k):

        doc_index = indices[0][i]
        similarity = 1 / (1 + distances[0][i])

        st.markdown(f"### Result {i+1}")
        st.write(f"Similarity Score: {similarity:.4f}")
        st.write(documents[doc_index][:300])
