import streamlit.components.v1 as components
import os
import io

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "st_audio_player",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("st_audio_player", path=build_dir)

def st_audio_player(audio_stream, key=None):
    for i, chunk in enumerate(audio_stream):
        chunk_str = chunk.getvalue().hex()
        _component_func(data=chunk_str, stream_active=True, key=f"audio_chunk_{i}")
    
    _component_func(data="", stream_active=False, key=f"audio_stream_complete_{key}")
