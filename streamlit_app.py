
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
import openai
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips  # type: ignore

# Configuration
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌  OpenAI API key not found – add it to secrets!")
    st.stop()

st.set_page_config(page_title="TL;DR Studios Generator", layout="wide")

# Session state
state_defaults = dict(
    script_text=None,
    audio_path=None,
    timestamps_json=None,
    storyboard_json=None,
    metadata_json=None,
    video_path=None,
)
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

@st.cache_data(show_spinner=False)
def save_tmp_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name

def call_openai_chat(prompt: str, system: str = "You are a helpful assistant") -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def call_openai_tts(text: str, voice: str = "alloy") -> bytes:
    resp = openai.audio.speech.create(
        model="tts-1",
        input=text,
        voice=voice,
        format="wav",
    )
    return resp.audio.data

def call_openai_whisper(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp.name,"rb"),
            response_format="text",
            timestamp_granularities=["segment"],
        )
    return transcript

# ----- Step functions (same as canvas, omitted here for brevity) -----
# For demonstration purposes, we'll keep rest identical to canvas code.
