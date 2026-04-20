import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🚀",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .result-card {
        background-color: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .score-badge {
        background-color: #e63946;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .result-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-top: 0.5rem;
    }
    .result-body {
        font-size: 0.9rem;
        color: #a0a0b0;
        margin-top: 0.4rem;
    }
    .explain-box {
        background-color: #1a1f35;
        border-left: 4px solid #e63946;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.88rem;
        color: #c0c0d0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Title ───────────────────────────────────────────────────────
st.title("🚀 Semantic Search Engine")
st.caption("FAISS + Sentence Embeddings | NLP Project Demo")

# ─── Load Model ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─── Load & Index Dataset ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading and indexing AG News dataset...")
def load_and_index(num_samples=5000):
    dataset = load_dataset("ag_news", split="train")
    texts = [item["text"] for item in dataset.select(range(num_samples))]

    model = load_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)   # Inner Product = cosine after normalization
    index.add(embeddings)

    return texts, index

texts, index = load_and_index()
model = load_model()

# ─── Search UI ───────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Enter your search query:", placeholder="e.g. space rockets, climate change, stock market...")

with col2:
    top_k = st.slider("Number of results:", min_value=1, max_value=10, value=3)

search_clicked = st.button("🔍 Search", use_container_width=False)

# ─── Explanation (always visible) ────────────────────────────────
with st.expander("💡 How does this work? (NLP Concepts)"):
    st.markdown("""
    1. **Sentence Embeddings** — Every text chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`, a transformer model. Semantically similar sentences have vectors that are *close together* in this high-dimensional space.
    2. **FAISS Index** — Facebook AI Similarity Search (FAISS) stores all vectors and enables ultra-fast nearest-neighbor lookups. We use cosine similarity (Inner Product after L2 normalization).
    3. **Query Encoding** — Your query is also embedded into the same vector space at search time.
    4. **Similarity Score** — The dot product between the query vector and document vectors gives a score between 0 and 1. Higher = more semantically relevant.
    
    **Why is this better than keyword search?**  
    A keyword search for *"space rockets"* would miss articles that say *"launch vehicle"* or *"propulsion system"*. Semantic search understands *meaning*, not just words.
    """)

# ─── Results ─────────────────────────────────────────────────────
if search_clicked and query.strip():
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    st.subheader(f"Results for: *{query}*")

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        text = texts[idx]
        # Split title (usually first sentence) from body
        parts = text.split(". ", 1)
        title = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""

        st.markdown(f"""
        <div class="result-card">
            <span class="score-badge">Score: {score:.4f}</span>
            <div class="result-title">Result {rank} — {title}</div>
            <div class="result-body">{body[:300]}{'...' if len(body) > 300 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    # Faculty-friendly explanation of the scores
    st.markdown(f"""
    <div class="explain-box">
        📊 <strong>Score Interpretation:</strong> Scores range from 0 to 1 (cosine similarity).
        Your top result scored <strong>{scores[0][0]:.4f}</strong> — 
        {'a strong semantic match.' if scores[0][0] > 0.5 else 'a moderate match — the model found related concepts even if exact words differ.'}
        Notice how results may not contain your exact query words — that's semantic search at work.
    </div>
    """, unsafe_allow_html=True)

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")

# ─── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption("Dataset: AG News (5000 samples) | Model: all-MiniLM-L6-v2 | Index: FAISS IndexFlatIP | NLP Project Demo")
