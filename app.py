import os
import torch
import tempfile
import streamlit as st
import whisper
from langdetect import detect
from googletrans import Translator

# Setup
os.environ["STREAMLIT_WATCH_SUPPRESS"] = "true"
st.set_page_config(page_title="Speech + Text Translator", layout="centered")
st.title("🌐 Hindi ↔ English Translator")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
@st.cache_resource
def load_models():
    stt_model = whisper.load_model("medium")  # Whisper Large
    translator = Translator()
    return stt_model, translator

whisper_model, translator = load_models()

# === Translate function ===
def translate(text, source_lang):
    target_lang = "hi" if source_lang == "en" else "en"
    result = translator.translate(text, src=source_lang, dest=target_lang)
    return result.text

# === Mode Selection ===
input_mode = st.radio("Choose input mode:", ["🎙️ Audio Upload", "⌨️ Manual Text"])

source_text = ""

# === Audio Upload Mode ===
if input_mode == "🎙️ Audio Upload":
    audio_file = st.file_uploader("📁 Upload Audio (.mp3 or .wav)", type=["mp3", "wav"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Transcribing with Whisper..."):
            result = whisper_model.transcribe(tmp_path)
            transcription = result["text"].strip()

        source_text = st.text_area("📝 Transcription (editable)", transcription, key="audio_text")

# === Manual Text Input Mode ===
elif input_mode == "⌨️ Manual Text":
    source_text = st.text_area("📝 Enter your text here (English or Hindi):", key="manual_text")

# === Translate if text is present ===
if source_text:
    try:
        lang_detected = detect(source_text)
        detected_lang = "hi" if lang_detected == "hi" else "en"
        st.info(f"🌐 Detected Language: {'Hindi' if detected_lang == 'hi' else 'English'}")
    except:
        detected_lang = "en"
        st.warning("⚠️ Language detection failed. Defaulting to English.")

    with st.spinner("🌍 Translating..."):
        translation = translate(source_text, detected_lang)

    st.success("✅ Translation Complete")
    st.text_area("🔁 Translated Output", translation, key="output_text")
