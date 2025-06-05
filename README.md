
# TL;DR Studios – Streamlit Video Generator

This project packages an end‑to‑end, **Streamlit‑powered** pipeline that turns a topic prompt into a finished explainer video using OpenAI’s APIs.

## Features
1. **Ideation & Script Generation**
2. **Text‑to‑Speech voice‑over**
3. **Whisper transcription with timestamps**
4. **AI‑generated storyboard prompts**
5. **SEO‑ready title, description, and thumbnail prompt**
6. **Draft video assembly** (MoviePy)

All intermediate outputs are cached, so you can hop between steps or upload your own files at any stage.

## Setup

```bash
git clone <your‑repo‑url>
cd tldr_studios_app
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Secrets

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

(When deploying on **Streamlit Cloud**, add the same key via *Settings → Secrets*.)

## Run locally

```bash
streamlit run streamlit_app.py
```

---

© 2025 TL;DR Studios
