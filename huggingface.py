
import streamlit as st
import pdfplumber

from transformers import pipeline
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load lightweight transformer pipelines
summary_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Utility Functions ---

def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    else:
        return file.read().decode("utf-8")

def generate_summary(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summary_pipe(chunk)[0]['summary_text'] for chunk in chunks[:3]]
    return " ".join(summaries)

def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    return vectordb, docs

def ask_question(question, context):
    result = qa_pipe(question=question, context=context)
    return result['answer']

def generate_questions(text):
    return [
        "What is the main objective of the research?",
        "What methods or techniques are discussed?",
        "What are the key findings or conclusions?"
    ]

def evaluate_answer(question, user_answer, context):
    # Dummy evaluation for simplicity
    if user_answer.lower() in context.lower():
        return "✅ Good answer! It matches the document context."
    else:
        return "⚠️ The answer doesn't clearly match the document. Try again."


# --- Streamlit UI ---

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("📚 GenAI Research Summarization Assistant")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"],
    key="file_uploader"
)

mode = st.radio("Select Mode", ["Auto Summary", "Ask Anything", "Challenge Me"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    text = extract_text(uploaded_file)

    if mode == "Auto Summary":
        summary = generate_summary(text)
        st.markdown("### 🔍 Summary")
        st.write(summary)

    elif mode == "Ask Anything":
        question = st.text_input("Ask your question:")
        if question:
            _, docs = create_vector_store(text)
            combined_context = " ".join([doc.page_content for doc in docs[:3]])
            answer = ask_question(question, combined_context)
            st.markdown("### 💬 Answer")
            st.write(answer)

    elif mode == "Challenge Me":
        st.markdown("### 🧠 Challenge Questions")
        questions = generate_questions(text)
        for i, q in enumerate(questions):
            user_ans = st.text_input(f"Q{i+1}: {q}", key=f"q{i}")
            if user_ans:
                feedback = evaluate_answer(q, user_ans, text)
                st.success(feedback)
