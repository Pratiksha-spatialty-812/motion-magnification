import streamlit as st
import cv2
import tempfile
import os
import time
from magnify import process_video

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motion Magnification Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #10101a !important;
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] * {
    color: #c8c8d8 !important;
}

/* Title area */
.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}

.sub-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 4px;
    letter-spacing: 0.05em;
}

/* Cards */
.info-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Parameter labels */
.param-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #a78bfa;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    width: 100%;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 2px dashed #2d2d4e !important;
    border-radius: 12px !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #38bdf8) !important;
}

/* Sliders */
[data-testid="stSlider"] .st-bd {
    background: #a78bfa !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 1rem;
}

/* Divider */
hr {
    border-color: #1e1e30 !important;
}

/* Success / info messages */
.stSuccess, .stInfo {
    border-radius: 10px !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-title">🔬 Motion Magnification Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Eulerian Video Magnification · Laplacian Pyramid · Temporal Filtering</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — Parameters ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("Tune the magnification settings below.")
    st.markdown("---")

    st.markdown('<div class="param-label">Alpha — Magnification Strength</div>', unsafe_allow_html=True)
    alpha = st.slider("Alpha", min_value=10, max_value=200, value=100, step=5,
                      help="Controls how much motion is amplified. Higher = more magnification.")

    st.markdown('<div class="param-label">Lambda C — Spatial Wavelength (px)</div>', unsafe_allow_html=True)
    lambda_c = st.slider("Lambda C", min_value=1, max_value=100, value=20, step=1,
                         help="Spatial frequency cutoff. Lower values magnify finer details.")

    st.markdown('<div class="param-label">fl — Low Frequency Cutoff (Hz)</div>', unsafe_allow_html=True)
    fl = st.slider("fl (Hz)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                   help="Lower bound of the temporal bandpass filter.")

    st.markdown('<div class="param-label">fh — High Frequency Cutoff (Hz)</div>', unsafe_allow_html=True)
    fh = st.slider("fh (Hz)", min_value=1.0, max_value=30.0, value=14.0, step=0.5,
                   help="Upper bound of the temporal bandpass filter.")

    st.markdown('<div class="param-label">FPS — Video Frame Rate</div>', unsafe_allow_html=True)
    fps = st.slider("FPS", min_value=15, max_value=60, value=30, step=1,
                    help="Frame rate of the input video.")

    st.markdown("---")
    st.markdown("### 📖 Parameter Guide")
    st.markdown("""
<div style='font-size:0.8rem; color:#6b7280; line-height:1.7;'>
<b style='color:#a78bfa'>Alpha:</b> Start at 100. Increase for subtle motions (heartbeat, breathing). Decrease if output looks noisy.<br><br>
<b style='color:#38bdf8'>fl / fh:</b> Set to match the frequency of motion you want to amplify. Heartbeat ≈ 1–2 Hz, breathing ≈ 0.5–1 Hz.<br><br>
<b style='color:#34d399'>Lambda C:</b> Lower values target finer spatial details. Start at 20.
</div>
""", unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Drop your video here",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supports MP4, AVI, MOV, MKV"
    )

    if uploaded_file:
        st.video(uploaded_file)

        # Video metadata
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_input_path = tmp.name

        cap = cv2.VideoCapture(tmp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / vid_fps if vid_fps > 0 else 0
        cap.release()

        st.markdown("#### 📊 Video Info")
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolution", f"{vid_w}×{vid_h}")
        m2.metric("FPS", f"{vid_fps:.0f}")
        m3.metric("Duration", f"{duration:.1f}s")

        st.markdown("#### ▶️ Run Processing")
        run_btn = st.button("🚀 Run Motion Magnification", use_container_width=True)
    else:
        tmp_input_path = None
        run_btn = False
        st.markdown("""
<div class="info-card">
<p style="color:#6b7280; font-size:0.9rem; margin:0;">
Upload a video on the left to get started. Tune parameters in the sidebar, then click <b style="color:#a78bfa">Run Motion Magnification</b>.
</p>
</div>
""", unsafe_allow_html=True)

with right_col:
    st.markdown("### 📥 Output")

    if uploaded_file and run_btn and tmp_input_path:
        output_path = tmp_input_path.replace(".mp4", "_magnified.mp4")

        status_box = st.empty()
        progress_bar = st.progress(0)
        eta_box = st.empty()

        status_box.info("⏳ Initialising magnification engine...")

        start_time = time.time()

        def progress_callback(current, total):
            pct = current / total
            progress_bar.progress(pct)
            elapsed = time.time() - start_time
            if current > 1 and pct > 0:
                eta = (elapsed / pct) * (1 - pct)
                eta_box.markdown(
                    f'<div style="font-family:Space Mono,monospace;font-size:0.75rem;color:#6b7280;">'
                    f'Frame {current}/{total} · ETA {eta:.0f}s remaining</div>',
                    unsafe_allow_html=True
                )

        try:
            status_box.info("🔄 Processing frames...")
            process_video(
                input_path=tmp_input_path,
                output_path=output_path,
                alpha=alpha,
                lambda_c=lambda_c,
                fl=fl,
                fh=fh,
                fps=fps,
                progress_callback=progress_callback,
            )
            progress_bar.progress(1.0)
            eta_box.empty()
            elapsed_total = time.time() - start_time
            status_box.success(f"✅ Done! Processed in {elapsed_total:.1f}s")

            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Magnified Video",
                    data=f,
                    file_name="magnified_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            # Clean up temp files
            try:
                os.remove(tmp_input_path)
            except Exception:
                pass

        except Exception as e:
            status_box.error(f"❌ Error: {str(e)}")
            progress_bar.empty()
            eta_box.empty()

    elif not uploaded_file:
        st.markdown("""
<div class="info-card" style="min-height:300px; display:flex; align-items:center; justify-content:center;">
<div style="text-align:center; color:#2d2d4e;">
    <div style="font-size:3rem;">🎬</div>
    <div style="font-family:'Space Mono',monospace; font-size:0.8rem; margin-top:0.5rem;">
        Output will appear here
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.7rem; color:#2d2d4e; padding: 1rem 0;">
    Eulerian Video Magnification · Based on MIT CSAIL Research · Built with Streamlit
</div>
""", unsafe_allow_html=True)
