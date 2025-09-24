import random
import numpy as np
import torch
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr
import spaces

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

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

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")

def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

def get_supported_languages_display() -> str:
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    return MODEL

try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model. Error: {e}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def chunk_text(text: str, max_len: int = 300):
    """Split text into manageable chunks (~sentences or max_len)."""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) <= max_len:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks

def resolve_audio_prompt(language_id: str, provided_path: str | None) -> str | None:
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")

@spaces.GPU
def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:

    current_model = get_or_load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)
    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt

    text_chunks = chunk_text(text_input, max_len=300)
    print(f"Splitting text into {len(text_chunks)} chunks.")

    audio_pieces = []
    for idx, chunk in enumerate(text_chunks):
        print(f"Generating chunk {idx+1}/{len(text_chunks)}: '{chunk[:50]}...' ")
        wav = current_model.generate(chunk, language_id=language_id, **generate_kwargs)
        audio_pieces.append(wav.squeeze(0).numpy())

    final_audio = np.concatenate(audio_pieces)
    print("Audio generation complete.")
    return (current_model.sr, final_audio)

with gr.Blocks() as demo:
    gr.Markdown("""
        # Chatterbox Multilingual Demo
        Generate high-quality multilingual speech from text with reference audio styling, supporting 23 languages.
        """)

    gr.Markdown(get_supported_languages_display())

    with gr.Row():
        with gr.Column():
            initial_lang = "fr"
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (no character limit)",
                max_lines=10
            )

            language_id = gr.Dropdown(
                choices=list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                value=initial_lang,
                label="Language"
            )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value=default_audio_for_ui(initial_lang)
            )

            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

        def on_language_change(lang, current_ref, current_text):
            return default_audio_for_ui(lang), default_text_for_ui(lang)

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text],
            outputs=[ref_wav, text],
            show_progress=False
        )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, language_id, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

demo.launch(mcp_server=True, share=True)
