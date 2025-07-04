# Genai-research-assistant
Smart AI Research Summarization Assistant
This repository contains three powerful versions of a Streamlit-based summarization and Q&A assistant for research papers and long documents.

Each version is tailored for different use cases:

> OpenAI-based: For GPT-3.5/4 powered cloud summarization and Q&A
> Ollama-based: For running local LLMs like Mistral or Phi-3
> Offline Transformers-based: 100% open-source, lightweight, and runs fully offline

// DRAWBACKS ;
 1. OpenAI-Based Assistant
Requires an API key with limited free quota.
Usage is subject to rate limits and billing depending on your OpenAI plan.
Requires a stable internet connection.
Not ideal for fully offline or privacy-sensitive use cases.

2. Ollama-Based Assistant
Requires high system resources:
At least 5–6 GB of RAM to run models like Mistral or LLaMA3.
Significant disk space (models can be 3–8 GB).
Performance is reduced on low-end CPUs or integrated graphics.
Initial model download can take time.

3. Offline Transformers-Based Assistant (Hugging Face)
Runs slowly, especially on CPUs without GPU acceleration.
Only lightweight models (like DistilBERT or Bart-small) are usable on low-RAM systems.
Not suitable for very long or complex documents.
Lacks the generative fluency of GPT or LLaMA-class models.

