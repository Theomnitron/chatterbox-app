import random
import numpy as np
import torch
import streamlit as st
import io
import torchaudio
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import re

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Chatterbox Multilingual TTS",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Chatterbox Multilingual TTS")
st.markdown(
    """
    <div style="text-align: center;">
        <p>A simple Streamlit interface for the Chatterbox Multilingual Text-to-Speech model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Global Model and Configuration ---
# Use st.cache_resource to cache the model, preventing it from reloading
# on every user interaction. This is the key to performance.
@st.cache_resource
def load_model():
    """Loads the Chatterbox model with a progress spinner."""
    with st.spinner("Loading model... This may take a moment."):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {DEVICE}")
        # The constructor needs these positional arguments.
        # Passing them as None is the most straightforward solution.
        model = ChatterboxMultilingualTTS(
            is_s3gen=False,
            vc=None,
            tokenizer=None,
            lang_name=None,
            device=DEVICE
        )
    return model

MODEL = load_model()

# --- UI Helpers ---
LANGUAGE_CONFIG = {
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    # Add other languages as needed...
}

def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")

def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

def get_supported_languages_display() -> str:
    language_items = []
    for lang_code, lang_name in sorted(list(SUPPORTED_LANGUAGES.items())):
        language_items.append(f"<li><b>{lang_name}</b> ({lang_code})</li>")
    return f"<ul>{''.join(language_items)}</ul>"

# --- Main App Logic ---
with st.sidebar:
    st.subheader("Model Parameters")
    
    # State management for default text and audio based on language
    if 'default_text' not in st.session_state:
        st.session_state.default_text = default_text_for_ui('en')
    if 'ref_audio_url' not in st.session_state:
        st.session_state.ref_audio_url = default_audio_for_ui('en')

    # Language selection
    language_id = st.selectbox(
        "Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda lang: SUPPORTED_LANGUAGES[lang],
        index=list(SUPPORTED_LANGUAGES.keys()).index('en'),
        on_change=lambda: st.session_state.update(
            default_text=default_text_for_ui(st.session_state.language_id),
            ref_audio_url=default_audio_for_ui(st.session_state.language_id)
        ),
        key='language_id'
    )

    # Reference Audio Upload
    uploaded_audio = st.file_uploader(
        "Upload Reference Audio File",
        type=['mp3', 'flac', 'wav'],
        help="Use a speaker's voice from this file for voice cloning.",
        label_visibility="visible"
    )

    # If an audio file is uploaded, use it. Otherwise, use the default URL.
    ref_wav_path = None
    if uploaded_audio:
        ref_wav_path = uploaded_audio
    elif st.session_state.ref_audio_url:
        ref_wav_path = st.session_state.ref_audio_url

    # Model generation parameters
    st.subheader("Generation Options")
    exaggeration = st.slider("Exaggeration", min_value=0.25, max_value=2.0, value=0.5, step=0.05)
    cfg_weight = st.slider("CFG/Pace", min_value=0.2, max_value=1.0, value=0.5, step=0.05)
    seed_num = st.number_input("Random Seed (0 for random)", min_value=0, value=0)
    temp = st.slider("Temperature", min_value=0.05, max_value=5.0, value=0.8, step=0.05)

# Text Input
text_area = st.text_area(
    "Enter the text to be converted to speech:",
    value=st.session_state.default_text,
    height=200
)

# Generate Button
if st.button("Generate Audio", use_container_width=True):
    if not text_area:
        st.warning("Please enter some text to convert.")
    else:
        # A simple generator to split text by sentences
        def text_chunk_generator(text):
            # Split text by sentences, keeping punctuation
            sentences = re.split(r'([.?!])', text)
            chunks = []
            current_chunk = ""
            for i, sentence in enumerate(sentences):
                current_chunk += sentence
                if sentence in ['.', '?', '!', '\n'] or i == len(sentences) - 1:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = ""
            return chunks

        text_chunks = text_chunk_generator(text_area)

        # Placeholder to display audio players dynamically
        audio_placeholder = st.empty()
        
        with st.spinner("Generating audio..."):
            for chunk in text_chunks:
                try:
                    # Generate the audio for the current chunk
                    generated_audio = MODEL.generate_audio(
                        text=chunk,
                        lang_id=language_id,
                        ref_wav=ref_wav_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        seed=seed_num if seed_num != 0 else random.randint(1, 10000),
                        temp=temp,
                    )

                    # Write the generated audio to an in-memory buffer
                    audio_buffer = io.BytesIO()
                    torchaudio.save(audio_buffer, generated_audio, MODEL.sampling_rate, format="wav")
                    audio_buffer.seek(0)
                    
                    # Display the audio player for the current chunk
                    st.audio(audio_buffer.getvalue(), format="audio/wav")
                    
                except Exception as e:
                    st.error(f"An error occurred during audio generation: {e}")
                    st.stop()
