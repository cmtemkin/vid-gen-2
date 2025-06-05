
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
import openai
from pydub import AudioSegment   # type: ignore
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips  # type: ignore

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌  OpenAI API key not found – add it to secrets!")
    st.stop()

st.set_page_config(page_title="TL;DR Studios Generator", layout="wide")

# Establish persistent state containers
state_defaults = dict(
    script_text     = None,
    audio_path      = None,
    timestamps_json = None,
    storyboard_json = None,
    metadata_json   = None,
    video_path      = None,
)
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def save_tmp_file(uploaded_file) -> str:
    """Persist an UploadedFile to a NamedTemporaryFile and return the path."""
    suffix = Path(uploaded_file.name).suffix
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded_file.getbuffer())
    tmp_file.close()
    return tmp_file.name

def call_openai_chat(prompt: str, system: str = "You are a helpful assistant") -> str:
    resp = openai.chat.completions.create(  # type: ignore[attr-defined]
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def call_openai_tts(text: str, voice: str = "alloy") -> bytes:
    resp = openai.audio.speech.create(  # type: ignore[attr-defined]
        model="tts-1",
        input=text,
        voice=voice,
        format="wav",
    )
    return resp.audio.data  # bytes

def call_openai_whisper(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio.flush()
        transcript = openai.audio.transcriptions.create(  # type: ignore[attr-defined]
            model="whisper-1",
            file=open(tmp_audio.name, "rb"),
            response_format="text",
            timestamp_granularities=["segment"],
        )
    return transcript

# ──────────────────────────────────────────────────────────────
# Pipeline step 1 – Ideation & Script Generation
# ──────────────────────────────────────────────────────────────

def step_generate_script():
    st.header("1 · Ideation & Script Generation")
    st.write("Provide a topic or full prompt. We'll return a video script.")

    prompt = st.text_area("Topic / prompt", placeholder="Explain why pigeons are low‑key cyberpunk icons")

    use_prev = st.button("🪄 Use previous script", disabled=not st.session_state.script_text)
    if use_prev:
        prompt = st.session_state.script_text or ""

    if st.button("Generate Script") and prompt:
        with st.spinner("Thinking…"):
            script_text = call_openai_chat(
                prompt,
                system=(
                    "You are an Emmy‑award‑winning explainer‑video writer. Return a tight, engaging, conversational script.\n\n"
                    "Respond with pure markdown/plain‑text – no code fences."),
            )
        st.session_state.script_text = script_text

    if st.session_state.script_text:
        st.subheader("📝 Script")
        st.markdown(st.session_state.script_text)

# ──────────────────────────────────────────────────────────────
# Pipeline step 2 – Audio Creation
# ──────────────────────────────────────────────────────────────

def step_audio_creation():
    st.header("2 · Audio Creation (TTS)")
    tab1, tab2 = st.tabs(["Generate", "Upload custom"])

    with tab1:
        if not st.session_state.script_text:
            st.info("Generate or upload a script first in Step 1.")
        else:
            voice = st.selectbox("Voice", ["alloy", "female", "male"], index=0)
            if st.button("Create voice‑over"):
                with st.spinner("Synthesizing…"):
                    audio_bytes = call_openai_tts(st.session_state.script_text, voice)
                    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    Path(audio_path).write_bytes(audio_bytes)
                    st.session_state.audio_path = audio_path
    with tab2:
        uploaded_audio = st.file_uploader("Upload a .wav/.mp3", type=["wav", "mp3", "m4a"], key="audio_up")
        if uploaded_audio:
            audio_path = save_tmp_file(uploaded_audio)
            st.session_state.audio_path = audio_path

    if st.session_state.audio_path:
        st.audio(st.session_state.audio_path)

# ──────────────────────────────────────────────────────────────
# Pipeline step 3 – Timestamped Transcription
# ──────────────────────────────────────────────────────────────

def step_transcription():
    st.header("3 · Timestamped Transcription")

    if not st.session_state.audio_path:
        st.info("Create or upload audio in Step 2 first.")

    else:
        if st.button("Transcribe & timestamp"):
            with st.spinner("Transcribing…"):
                transcript = call_openai_whisper(Path(st.session_state.audio_path).read_bytes())
                # simple segment split by \n
                segments = [
                    {"start": i*5, "end": (i+1)*5, "text": line.strip()}
                    for i, line in enumerate(transcript.splitlines()) if line.strip()
                ]
                st.session_state.timestamps_json = segments

    if st.session_state.timestamps_json:
        st.json(st.session_state.timestamps_json)

# ──────────────────────────────────────────────────────────────
# Pipeline step 4 – Storyboard Creation
# ──────────────────────────────────────────────────────────────

def step_storyboard():
    st.header("4 · Storyboard Creation")

    if not st.session_state.timestamps_json:
        st.info("Complete transcription (Step 3) first.")
        return

    if st.button("Generate storyboard prompts"):
        with st.spinner("Drafting…"):
            sb_prompts = []
            for seg in st.session_state.timestamps_json:
                prompt = call_openai_chat(
                    f"Create an image prompt for a video scene: {seg['text']}",
                    system="You are a creative visual prompt generator for DALL‑E 3.")
                sb_prompts.append({**seg, "prompt": prompt})
            st.session_state.storyboard_json = sb_prompts

    if st.session_state.storyboard_json:
        st.json(st.session_state.storyboard_json)

# ──────────────────────────────────────────────────────────────
# Pipeline step 5 – Title, Description & Cover
# ──────────────────────────────────────────────────────────────

def step_metadata():
    st.header("5 · Title, Description & Cover")
    if not st.session_state.script_text:
        st.info("Generate a script first (Step 1).")

    else:
        if st.button("Generate metadata"):
            with st.spinner("Copywriting…"):
                meta_resp = call_openai_chat(
                    st.session_state.script_text,
                    system="You are a YouTube strategist – craft a catchy title,\nSEO description (≤200 words) and a DALL‑E prompt for the thumbnail.")
                st.session_state.metadata_json = meta_resp
    if st.session_state.metadata_json:
        st.markdown(st.session_state.metadata_json)

# ──────────────────────────────────────────────────────────────
# Pipeline step 6 – Video Assembly
# ──────────────────────────────────────────────────────────────

def step_video():
    st.header("6 · Video Assembly (beta)")
    if not (st.session_state.storyboard_json and st.session_state.audio_path):
        st.info("Need storyboard (Step 4) and audio (Step 2).")
        return

    if st.button("Assemble draft video"):
        with st.spinner("Rendering… this can take a minute"):
            imgs: List[ImageClip] = []
            duration_per_slide = 5
            for scene in st.session_state.storyboard_json:
                # placeholder: blank color clip with text overlay
                img_clip = ImageClip("black", duration=duration_per_slide).fx(
                    lambda c: c.txt_clip(scene.get("prompt", "Scene")))
                imgs.append(img_clip)
            audio_clip = AudioFileClip(st.session_state.audio_path)
            video = concatenate_videoclips(imgs).set_audio(audio_clip)
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            video.write_videofile(out_path, fps=24)
            st.session_state.video_path = out_path

    if st.session_state.video_path:
        st.video(st.session_state.video_path)

# ──────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────
pages = {
    "Script": step_generate_script,
    "Audio": step_audio_creation,
    "Transcription": step_transcription,
    "Storyboard": step_storyboard,
    "Metadata": step_metadata,
    "Video": step_video,
}
choice = st.sidebar.radio("Pipeline stage", list(pages.keys()))
pages[choice]()
