import streamlit as st

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("📚 GenAI Research Summarization Assistant")

uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

mode = st.radio("Select Mode", ["Auto Summary", "Ask Anything", "Challenge Me"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    # Processing logic here

import pdfplumber

def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    else:
        return file.read().decode("utf-8")


import openai

def generate_summary(text):
    prompt = f"Summarize this research article in 150 words:\n{text[:4000]}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    return vectordb

#
def ask_question(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4"),
        retriever=retriever,
        return_source_documents=True
    )
    return chain.run(question)

 #
def generate_questions(text):
    prompt = f"Based on the following research text, generate 3 logic-based or comprehension-focused questions:\n\n{text[:2000]}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

#

def evaluate_answer(question, user_answer, text):
    prompt = f"""Evaluate the user's answer to this question based on the document:
Question: {question}
User Answer: {user_answer}
Document: {text[:3000]}
Respond with correctness and justification."""
    ...

