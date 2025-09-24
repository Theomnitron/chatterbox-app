import random
import numpy as np
import torch
import streamlit as st
import io
import torchaudio
from pathlib import Path
import tempfile
import soundfile as sf

# The custom audio player component
from st_audio_player import st_audio_player

# Assuming src.chatterbox is in the Python path or a local package.
# It's important that `pip install -e src/` worked or that the chatterbox
# library is installed correctly for this to run.
# from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

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
        # Placeholder for your actual model loading code
        # model = ChatterboxMultilingualTTS(device=DEVICE)
        # model.load_model()
        return "Dummy Model"

MODEL = load_model()

# --- Placeholder for streaming generation ---
def get_audio_stream(model, text: str, lang_id: str, ref_wav_path=None, exaggeration=0.5, cfg_weight=0.5, seed=0, temp=0.8):
    """
    Generates audio in a streaming fashion.
    This function simulates the streaming process. You need to replace the
    logic with your model's streaming capabilities.
    """
    # Placeholder for a sentence splitter. Use a more robust library like NLTK
    sentences = text.replace('.', '.<SENTENCE_BREAK>').replace('?', '?<SENTENCE_BREAK>').replace('!', '!<SENTENCE_BREAK>').split('<SENTENCE_BREAK>')

    # Convert the uploaded file to a temporary WAV file for the model
    temp_ref_wav = None
    if ref_wav_path:
        temp_ref_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_ref_wav_file.name, ref_wav_path, 44100, format='WAV') # Assuming a default sample rate
        temp_ref_wav = temp_ref_wav_file.name

    try:
        # Loop through each sentence and yield the audio chunk
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # This is where your actual streaming model call would go.
            # You would call a function like `model.stream_audio(sentence, ...)`
            # and it would return a generator or stream.
            
            # --- Placeholder for the model's streaming logic ---
            # simulated_audio_chunk = model.generate_streaming_chunk(
            #     text=sentence,
            #     lang_id=lang_id,
            #     ref_wav=temp_ref_wav,
            #     exaggeration=exaggeration,
            #     cfg_weight=cfg_weight,
            #     seed=seed,
            #     temp=temp
            # )
            
            # For demonstration, we'll generate a random chunk
            sample_rate = 22050
            duration = len(sentence) / 10.0
            audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)

            # Convert numpy array to BytesIO for streaming
            audio_io = io.BytesIO()
            torchaudio.save(audio_io, torch.tensor(audio_data).unsqueeze(0), sample_rate, format="wav")
            audio_io.seek(0)
            yield audio_io
    finally:
        # Clean up the temporary file
        if temp_ref_wav:
            Path(temp_ref_wav).unlink()

# --- UI Components ---
with st.container():
    col1, col2 = st.columns([1, 1])

    with col1:
        text = st.text_area(
            "Enter your text:",
            "Welcome to the new era of real-time audio generation. This application streams the voice to you as it's being generated. No more waiting for the full file to be ready.",
            height=300,
            key="text_input"
        )
        
        language_id = st.selectbox(
            "Language",
            ["en", "fr"], # Replace with SUPPORTED_LANGUAGES.keys()
            key="language_id"
        )

        uploaded_audio = st.file_uploader(
            "Upload reference audio for voice cloning (optional):",
            type=["wav", "mp3", "flac"],
            key="uploaded_audio"
        )
    
    with col2:
        exaggeration = st.slider("Exaggeration", 0.25, 2.0, step=0.05, value=0.5, key="exaggeration")
        cfg_weight = st.slider("CFG/Pace", 0.2, 1.0, step=0.05, value=0.5, key="cfg_weight")
        seed_num = st.number_input("Random seed (0 for random)", value=0, key="seed_num")
        temp = st.slider("Temperature", 0.05, 5.0, step=0.05, value=0.8, key="temp")

# The generate button
if st.button("Generate Audio", use_container_width=True, type="primary"):
    if not text:
        st.error("Please enter some text to generate audio.")
    else:
        # Use st.status to provide feedback to the user
        with st.status("Generating audio...", expanded=True) as status:
            try:
                # Prepare the reference audio if uploaded
                ref_wav_path = None
                if uploaded_audio:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(uploaded_audio.getvalue())
                        ref_wav_path = temp_file.name

                status.update(label="Streaming audio to player...", state="running", expanded=True)

                # Get the audio stream from the generator
                audio_stream = get_audio_stream(
                    MODEL,
                    text,
                    language_id,
                    ref_wav_path,
                    exaggeration,
                    cfg_weight,
                    seed_num if seed_num != 0 else random.randint(1, 10000),
                    temp
                )
                
                # Pass the stream to the custom component in the sidebar
                with st.sidebar:
                    st.header("Audio Player")
                    st_audio_player(audio_stream)

                status.update(label="Audio stream complete!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"An error occurred during audio generation: {e}")
                status.update(label="Audio generation failed!", state="error", expanded=True)
