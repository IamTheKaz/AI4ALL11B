from streamlit_webrtc import webrtc_streamer
import streamlit as st

st.title("WebRTC Live Stream Test")
webrtc_streamer(
    key="example",
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)