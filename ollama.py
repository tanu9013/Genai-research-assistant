import streamlit as st
import pdfplumber

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Utility Functions ---

def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    else:
        return file.read().decode("utf-8")

def generate_summary(text):
    prompt = PromptTemplate.from_template("Summarize this research article in 150 words:\n{input_text}")
    chain = LLMChain(llm=Ollama(model="mistral"), prompt=prompt)
    return chain.run(input_text=text[:1500])

def generate_questions(text):
    prompt = PromptTemplate.from_template(
        "Generate 3 logic-based or comprehension-focused questions based on the following document:\n{input_text}"
    )
    chain = LLMChain(llm=Ollama(model="mistral"), prompt=prompt)
    return chain.run(input_text=text[:1500])

def evaluate_answer(question, user_answer, text):
    full_prompt = f"""Evaluate this answer based on the document.
Question: {question}
Answer: {user_answer}
Document: {text[:1500]}
Provide detailed feedback."""
    llm = Ollama(model="mistral")
    return llm(full_prompt)

def ask_question(question, text):
    full_prompt = f"""Document: {text[:2000]}

Answer this question based on the document:
{question}"""
    llm = Ollama(model="mistral")
    return llm(full_prompt)

# --- Streamlit UI ---

st.set_page_config(page_title="Research Assistant (Ollama)", layout="wide")
st.title("📚 GenAI Research Summarization Assistant (Ollama)")

uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
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
            answer = ask_question(question, text)
            st.markdown("### 💬 Answer")
            st.write(answer)

    elif mode == "Challenge Me":
        st.markdown("### 🧠 Challenge Questions")
        questions = generate_questions(text).split("\n")
        for i, q in enumerate(questions):
            user_ans = st.text_input(f"Q{i+1}: {q}", key=f"q{i}")
            if user_ans:
                feedback = evaluate_answer(q, user_ans, text)
                st.success(feedback)
