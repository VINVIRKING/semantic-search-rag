from transformers import pipeline
import streamlit as st
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

st.title("🚀 Semantic Search Engine (FAISS + Embeddings)")

@st.cache_resource
def load_model_and_data():
    dataset = load_dataset("ag_news", split="train[:1000]")
    documents = dataset["text"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return model, index, documents, generator


model, index, documents, generator = load_model_and_data()

query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results:", 1, 5, 3)

if st.button("Search") and query:
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    st.subheader("Results:")

    for i in range(top_k):
        doc_index = indices[0][i]
        similarity = 1 / (1 + distances[0][i])

        st.markdown(f"### Result {i+1}")
        st.write(f"Similarity Score: {similarity:.4f}")
        st.write(documents[doc_index][:300])
