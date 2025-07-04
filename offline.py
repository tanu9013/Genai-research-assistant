import streamlit as st
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Load lightweight models ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utility Functions ---
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    else:
        return file.read().decode("utf-8")

def generate_summary(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks[:2]]
    return " ".join(summaries)

def build_faiss_index(text):
    paragraphs = [p for p in text.split("\n") if len(p.strip()) > 40]
    embeddings = embedder.encode(paragraphs)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index, paragraphs

def answer_question(question, text, index, paragraphs):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed), k=3)
    context = " ".join(paragraphs[i] for i in I[0])
    result = qa_model(question=question, context=context)
    return result["answer"], context

def generate_challenge_questions(text):
    prompts = [
        f"What is the main conclusion of the text below?\n{text[:400]}",
        f"What does the author imply in this passage?\n{text[400:800]}",
        f"What is the evidence for the central claim?\n{text[800:1200]}"
    ]
    return prompts

# --- UI ---
st.set_page_config(page_title="Offline Research Assistant", layout="wide")
st.title("📚 Offline Research Assistant (Transformers + FAISS)")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
mode = st.radio("Choose Mode", ["Auto Summary", "Ask Anything", "Challenge Me"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    text = extract_text(uploaded_file)

    if mode == "Auto Summary":
        st.markdown("### 🔍 Summary")
        summary = generate_summary(text)
        st.write(summary)

    elif mode == "Ask Anything":
        st.markdown("### 💬 Ask a Question")
        question = st.text_input("Your question:")
        if question:
            index, paragraphs = build_faiss_index(text)
            answer, context = answer_question(question, text, index, paragraphs)
            st.write(f"**Answer:** {answer}")
            with st.expander("Show supporting context"):
                st.write(context)

    elif mode == "Challenge Me":
        st.markdown("### 🧠 Challenge Questions")
        questions = generate_challenge_questions(text)
        for i, q in enumerate(questions):
            user_ans = st.text_input(f"Q{i+1}: {q}", key=f"q{i}")
            if user_ans:
                st.success("✅ Answer submitted! (Offline version does not evaluate answers)")
