import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("🚀 WebRTC Live Stream Test")

webrtc_streamer(
    key="live_stream",
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
