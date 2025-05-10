import whisper
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load Whisper model (medium) for STT ===
print("🔊 Loading Whisper model...")
whisper_model = whisper.load_model("medium")

# === Load Meta's NLLB-200 distilled model for translation ===
print("🌐 Loading NLLB-200 model...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set source and target language codes
src_lang = "hin_Deva"  # Hindi (Devanagari)
tgt_lang = "eng_Latn"  # English (Latin)

# === Step 1: Transcribe Hindi audio ===
audio_path = "sample_hindi.mp3"  # Ensure this file is in your project folder
print("🎙️ Transcribing audio...")
result = whisper_model.transcribe(audio_path, language="hi")
hindi_text = result["text"]
print(f"\n🗣️ Hindi Transcription:\n{hindi_text}")

# === Step 2: Translate sentence-by-sentence using NLLB ===
print("\n🌍 Translating to English...")
tokenizer.src_lang = src_lang
sentences = re.split(r'[।.!?]', hindi_text)
translated_sentences = []

for sentence in sentences:
    sentence = sentence.strip()
    if sentence:
        # Add target language token to help NLLB translate correctly
        sentence = f'>>{tgt_lang}<< {sentence}'
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(f'>>{tgt_lang}<<')
        )
        english = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        translated_sentences.append(english)

# Join translated sentences into final text
english_text = " ".join(translated_sentences)

# === Final Output ===
print(f"\n🗣️ Final English Translation:\n{english_text}")
