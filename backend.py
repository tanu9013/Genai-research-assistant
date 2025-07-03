# backend.py
import openai

def ask_question(question, text):
    context = text[:3000]  # use top chunks only
    prompt = f"Answer the question based on the context below. Include the supporting sentence.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    full_text = response['choices'][0]['message']['content']
    if "Reference:" in full_text:
        answer, ref = full_text.split("Reference:", 1)
        return answer.strip(), ref.strip()
    return full_text.strip(), "N/A"

def generate_challenges(text):
    prompt = f"Generate 3 logic-based or comprehension-focused questions from this document:\n{text[:3000]}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip().split("\n")

def evaluate_response(question, user_answer, text):
    prompt = f"""
Evaluate the following answer against the document.

Question: {question}
User Answer: {user_answer}

Document: {text[:3000]}

Provide a 0/1 score and a short justification.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return 1 if "correct" in response['choices'][0]['message']['content'].lower() else 0, \
           response['choices'][0]['message']['content'].strip()
