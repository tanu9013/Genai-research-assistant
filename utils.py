# utils.py
import pdfplumber

def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    else:
        return file.read().decode("utf-8")

def generate_summary(text):
    import openai
    prompt = f"Summarize this research paper in 150 words:\n\n{text[:4000]}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']
