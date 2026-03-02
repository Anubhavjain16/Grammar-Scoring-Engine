import io
import os
import time
import wave
from queue import Empty

import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Real-Time English Communication Assessment", layout="wide")
st.title("🎙️ Real-Time English Communication Assessment")
st.caption("SHL/Versant-style spoken grammar evaluation powered by Whisper + Transformer grammar scoring")

left, right = st.columns([2, 1])

with right:
    st.subheader("Session Metrics")
    transcript_box = st.empty()
    grammar_box = st.metric("Grammar Score (0–5)", "-")
    confidence_box = st.metric("Confidence", "-")


def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int = 48000) -> bytes:
    samples = samples.astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()


with left:
    st.subheader("Mic Stream")
    st.write("Click START and speak naturally. Results update every few seconds.")

    ctx = webrtc_streamer(
        key="speech-assessment",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
    )

    analyze_btn = st.button("Analyze Latest Chunk")

if analyze_btn:
    if not ctx.audio_receiver:
        st.error("Audio receiver not ready. Start microphone first.")
    else:
        all_chunks = []
        for _ in range(15):
            try:
                frames = ctx.audio_receiver.get_frames(timeout=0.2)
            except Empty:
                frames = []
            for frame in frames:
                arr = frame.to_ndarray()
                mono = arr[0] if arr.ndim > 1 else arr
                all_chunks.append(mono)
            time.sleep(0.05)

        if not all_chunks:
            st.warning("No audio captured yet. Please speak and retry.")
        else:
            audio = np.concatenate(all_chunks)
            wav_bytes = pcm_to_wav_bytes(audio)
            files = {"audio": ("speech.wav", wav_bytes, "audio/wav")}

            try:
                with st.spinner("Analyzing speech..."):
                    response = requests.post(f"{BACKEND_URL}/analyze", files=files, timeout=120)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Backend connection/error: {exc}")
            else:
                payload = response.json()
                transcript_box.text_area("Transcript", payload.get("transcript", ""), height=220)
                grammar_box.metric("Grammar Score (0–5)", payload.get("grammar_score", "-"))
                confidence_box.metric("Confidence", payload.get("grammar_confidence", "-"))
                st.success("Analysis complete")
