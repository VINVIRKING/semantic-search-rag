# Semantic Search RAG Engine

A semantic search engine powered by **Sentence Transformers, FAISS, and Retrieval-Augmented Generation (RAG)** deployed using **Streamlit**.

## Features

- Semantic search using transformer embeddings
- Fast similarity search using FAISS
- AI-generated answers using HuggingFace LLM
- Interactive web interface with Streamlit

## Tech Stack

- Python
- Sentence Transformers
- FAISS Vector Database
- HuggingFace Transformers
- Streamlit

## How It Works

1. Convert documents into embeddings
2. Store embeddings in FAISS index
3. Convert user query into embedding
4. Retrieve most relevant documents
5. Generate answer using LLM

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
