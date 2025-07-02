# app.py
import streamlit as st
# Custom CSS styling
st.markdown("""
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1602526216034-766f5fa4d39d');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Heading Styling */
    h1 {
        color: #f1f1f1;
        text-shadow: 2px 2px #000;
    }

    /* Radio Buttons */
    div[data-baseweb="radio"] > div {
        background-color: #ffffff10;
        border-radius: 10px;
        padding: 1em;
    }

    /* File uploader box */
    section[data-testid="stFileUploader"] {
        background-color: #ffffffaa;
        padding: 1em;
        border-radius: 10px;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #eeeeff;
        border: 2px solid #8888ff;
        border-radius: 10px;
        padding: 0.5em;
        font-size: 16px;
    }

    /* Markdowns (Answers and Summary) */
    .stMarkdown {
        background-color: #ffffffbb;
        border-radius: 10px;
        padding: 1em;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("📘 GenAI Research Summarization Assistant")

uploaded_file = st.file_uploader("Upload your research document", type=["pdf", "txt"])

if uploaded_file:
    # Choose interaction mode
    mode = st.radio("Choose Interaction Mode", ["Auto Summary", "Ask Anything", "Challenge Me"])

    if "history" not in st.session_state:
        st.session_state.history = []

    # Process text
    from utils import extract_text, generate_summary
    text = extract_text(uploaded_file)

    if mode == "Auto Summary":
        summary = generate_summary(text)
        st.markdown("### 🔍 Summary")
        st.write(summary)

    elif mode == "Ask Anything":
        from backend import ask_question
        question = st.text_input("Ask a question about the document")
        if question:
            answer, snippet = ask_question(question, text)
            st.markdown("**Answer:** " + answer)
            st.markdown("> 📌 Reference: `" + snippet + "`")
            st.session_state.history.append((question, answer))

    elif mode == "Challenge Me":
        from backend import generate_challenges, evaluate_response
        questions = generate_challenges(text)
        for q in questions:
            user_ans = st.text_input(f"🧠 {q}", key=q)
            if user_ans:
                score, feedback = evaluate_response(q, user_ans, text)
                st.markdown(f"✅ Score: {score}/1\n\n💬 Feedback: {feedback}")
