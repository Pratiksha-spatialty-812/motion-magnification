import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import os
import io
import time
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from magnify import process_video, analyze_vibration, ALPHA_CURVES

st.set_page_config(
    page_title="Motion Magnification Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e8f0; }
[data-testid="stSidebar"] { background-color: #10101a !important; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }
.main-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0; line-height: 1.1;
}
.sub-title { font-family: 'Space Mono', monospace; font-size: 0.82rem; color: #6b7280; margin-top: 4px; }
.step-badge {
    display: inline-block; font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #2563eb); color: #fff;
    border-radius: 20px; padding: 3px 14px; letter-spacing: 0.06em; margin-bottom: 8px;
}
.step-badge.done   { background: linear-gradient(135deg, #059669, #0284c7); }
.step-badge.locked { background: #1e1e30; color: #4b5563; }
.param-label { font-family: 'Space Mono', monospace; font-size: 0.73rem; color: #a78bfa; text-transform: uppercase; margin-bottom: 2px; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; width: 100%;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0284c7) !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; width: 100%;
}
.stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #38bdf8) !important; }
[data-testid="stMetric"] { background: #13131f; border: 1px solid #1e1e30; border-radius: 10px; padding: 1rem; }
hr { border-color: #1e1e30 !important; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 Motion Magnification Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Eulerian Video Magnification · Laplacian Pyramid · Temporal Filtering · Vibration Analysis</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("magnified_path", None),
    ("mag_vid_w",      0),
    ("mag_vid_h",      0),
    ("mag_vid_fps",    0.0),
    ("confirmed_roi",  None),   # (x,y,w,h) — only written by Confirm button
    ("orig_tmp_path",  None),   # saved tmp path for original
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Magnification Parameters")
    st.markdown("---")
    st.markdown('<div class="param-label">Alpha</div>', unsafe_allow_html=True)
    alpha = st.slider("Alpha", 10, 200, 100, 5)
    st.markdown('<div class="param-label">Lambda C (px)</div>', unsafe_allow_html=True)
    lambda_c = st.slider("Lambda C", 1, 100, 20, 1)
    st.markdown('<div class="param-label">fl — Low Cutoff (Hz)</div>', unsafe_allow_html=True)
    fl = st.slider("fl (Hz)", 0.1, 10.0, 1.0, 0.1)
    st.markdown('<div class="param-label">fh — High Cutoff (Hz)</div>', unsafe_allow_html=True)
    fh = st.slider("fh (Hz)", 1.0, 30.0, 14.0, 0.5)
    st.markdown('<div class="param-label">FPS</div>', unsafe_allow_html=True)
    fps_sidebar = st.slider("FPS", 15, 60, 30, 1)
    st.markdown("---")
    st.markdown("### 🔺 Pyramid Settings")
    n_levels_raw = st.slider("Pyramid Levels (0=auto)", 0, 8, 0, 1)
    n_levels = None if n_levels_raw == 0 else n_levels_raw
    alpha_curve = st.selectbox("Alpha Curve", list(ALPHA_CURVES.keys()), index=0)
    st.markdown("---")
    st.markdown("### 🌊 Vibration Analysis")
    of_method = st.selectbox("Optical Flow Method", ["farneback", "lucas_kanade"])

# ── Vibration plot ─────────────────────────────────────────────────────────────
def make_plot(result):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), facecolor="#0d0d18", constrained_layout=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d18")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e1e30")
    ax1.plot(result["times"], result["motion"], color="#38bdf8", lw=1.2)
    ax1.set_xlabel("Time (s)", color="#9ca3af", fontsize=8)
    ax1.set_ylabel("Mean Flow (px/frame)", color="#9ca3af", fontsize=8)
    ax1.set_title("Motion Signal", color="#e8e8f0", fontsize=10, fontweight="bold")
    ax1.grid(color="#1e1e30", linestyle="--", linewidth=0.5)
    freqs, power = result["freqs"], result["power"]
    ax2.fill_between(freqs, power, alpha=0.3, color="#a78bfa")
    ax2.plot(freqs, power, color="#a78bfa", lw=1.2)
    dom = result["dominant_hz"]
    ax2.axvline(dom, color="#f59e0b", lw=1.5, linestyle="--", label=f"Peak: {dom:.3f} Hz")
    ax2.set_xlabel("Frequency (Hz)", color="#9ca3af", fontsize=8)
    ax2.set_ylabel("Power", color="#9ca3af", fontsize=8)
    ax2.set_title("FFT Vibration Spectrum", color="#e8e8f0", fontsize=10, fontweight="bold")
    ax2.legend(facecolor="#13131f", edgecolor="#1e1e30", labelcolor="#f59e0b", fontsize=8)
    ax2.grid(color="#1e1e30", linestyle="--", linewidth=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Upload & Magnify
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">STEP 1 — Upload &amp; Magnify</div>', unsafe_allow_html=True)
st.markdown("### 📤 Upload Video")

uploaded_file = st.file_uploader("Drop your video here", type=["mp4", "avi", "mov", "mkv"])

tmp_input_path = None
vid_w = vid_h = total_frames = 0
vid_fps = 0.0

if uploaded_file:
    # Write to a stable tmp file once per upload
    if st.session_state["orig_tmp_path"] is None or \
       not os.path.exists(st.session_state["orig_tmp_path"]):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state["orig_tmp_path"] = tmp.name
    tmp_input_path = st.session_state["orig_tmp_path"]

    cap = cv2.VideoCapture(tmp_input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps  = cap.get(cv2.CAP_PROP_FPS)
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / vid_fps if vid_fps > 0 else 0

    left_up, right_up = st.columns([1, 1], gap="large")

    with left_up:
        st.markdown("**Original**")
        # st.video reads from the UploadedFile object — seek to start first
        uploaded_file.seek(0)
        st.video(uploaded_file)
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolution", f"{vid_w}×{vid_h}")
        m2.metric("FPS", f"{vid_fps:.0f}")
        m3.metric("Duration", f"{duration:.1f}s")

    with right_up:
        st.markdown("**Magnified output**")
        run_btn = st.button("🚀 Run Motion Magnification", use_container_width=True)

        if run_btn:
            st.session_state["magnified_path"] = None
            st.session_state["confirmed_roi"]  = None

            output_path  = tmp_input_path.replace(".mp4", "_magnified.mp4")
            status_box   = st.empty()
            progress_bar = st.progress(0)
            eta_box      = st.empty()
            t0 = time.time()

            def _mag_cb(cur, tot):
                progress_bar.progress(cur / tot)
                elapsed = time.time() - t0
                if cur > 1:
                    eta = (elapsed / (cur/tot)) * (1 - cur/tot)
                    eta_box.caption(f"Frame {cur}/{tot} · ETA {eta:.0f}s")

            try:
                status_box.info("🔄 Processing frames…")
                out_path = process_video(
                    input_path=tmp_input_path, output_path=output_path,
                    alpha=alpha, lambda_c=lambda_c, fl=fl, fh=fh, fps=fps_sidebar,
                    progress_callback=_mag_cb, n_levels=n_levels, alpha_curve=alpha_curve,
                )
                progress_bar.progress(1.0); eta_box.empty()
                status_box.success(f"✅ Done in {time.time()-t0:.1f}s")
                st.session_state["magnified_path"] = out_path
                st.session_state["mag_vid_w"]   = vid_w
                st.session_state["mag_vid_h"]   = vid_h
                st.session_state["mag_vid_fps"] = vid_fps if vid_fps > 0 else float(fps_sidebar)
            except Exception as e:
                status_box.error(f"❌ Error: {e}")

        # Show magnified video — read from file path stored in session_state
        mag_path_ss = st.session_state.get("magnified_path")
        if mag_path_ss and os.path.exists(mag_path_ss):
            with open(mag_path_ss, "rb") as f:
                mag_bytes = f.read()
            st.video(mag_bytes, format="video/mp4")
            st.download_button(
                "⬇️ Download Magnified Video", data=mag_bytes,
                file_name="magnified_output.mp4", mime="video/mp4",
                use_container_width=True,
            )

else:
    st.session_state["orig_tmp_path"] = None
    st.session_state["magnified_path"] = None
    st.session_state["confirmed_roi"]  = None
    st.info("Upload a video above, tune parameters in the sidebar, then click **Run Motion Magnification**.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ROI picker (pure Streamlit — no JS iframe bridge)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

mag_path  = st.session_state.get("magnified_path")
mag_ready = bool(mag_path and os.path.exists(mag_path))

if not mag_ready:
    st.markdown('<div class="step-badge locked">STEP 2 — Draw ROI · run magnification first</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-badge locked" style="margin-top:6px">STEP 3 — Vibration Analysis · run magnification first</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="step-badge done">STEP 2 — Select ROI on Magnified Frame</div>', unsafe_allow_html=True)

    m_w   = st.session_state["mag_vid_w"]
    m_h   = st.session_state["mag_vid_h"]
    m_fps = st.session_state["mag_vid_fps"]

    cap_m = cv2.VideoCapture(mag_path)
    ret_m, first_frame = cap_m.read()
    cap_m.release()

    if not ret_m:
        st.warning("Could not read first frame of magnified video.")
    else:
        st.markdown(
            "Set **X, Y, Width, Height** of the region to analyse. "
            "The preview below updates live."
        )

        # ── Numeric ROI inputs — pure Streamlit, no JS needed ─────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="param-label">X (left)</div>', unsafe_allow_html=True)
            roi_x = st.number_input("X", min_value=0, max_value=max(m_w-1, 0),
                                    value=int(m_w * 0.25), step=1, label_visibility="collapsed")
        with c2:
            st.markdown('<div class="param-label">Y (top)</div>', unsafe_allow_html=True)
            roi_y = st.number_input("Y", min_value=0, max_value=max(m_h-1, 0),
                                    value=int(m_h * 0.25), step=1, label_visibility="collapsed")
        with c3:
            st.markdown('<div class="param-label">Width</div>', unsafe_allow_html=True)
            roi_w = st.number_input("W", min_value=4, max_value=max(m_w, 4),
                                    value=int(m_w * 0.5), step=1, label_visibility="collapsed")
        with c4:
            st.markdown('<div class="param-label">Height</div>', unsafe_allow_html=True)
            roi_h = st.number_input("H", min_value=4, max_value=max(m_h, 4),
                                    value=int(m_h * 0.5), step=1, label_visibility="collapsed")

        # Clamp to frame bounds
        roi_x = int(min(roi_x, m_w - 4))
        roi_y = int(min(roi_y, m_h - 4))
        roi_w = int(min(roi_w, m_w - roi_x))
        roi_h = int(min(roi_h, m_h - roi_y))

        # Live preview with rectangle
        preview = first_frame.copy()
        cv2.rectangle(preview, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (245, 158, 11), 2)
        # Overlay semi-transparent fill
        overlay = preview.copy()
        cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (245, 158, 11), -1)
        preview = cv2.addWeighted(overlay, 0.15, preview, 0.85, 0)
        cv2.rectangle(preview, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (245, 158, 11), 2)

        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                 caption=f"ROI preview — x:{roi_x}  y:{roi_y}  w:{roi_w}  h:{roi_h}",
                 use_container_width=True)

        if st.button("✅ Confirm ROI", use_container_width=True):
            st.session_state["confirmed_roi"] = (roi_x, roi_y, roi_w, roi_h)
            st.success(f"ROI confirmed: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

        # Show currently confirmed ROI
        confirmed = st.session_state.get("confirmed_roi")
        if confirmed:
            st.info(f"✅ Active ROI for analysis: x={confirmed[0]}, y={confirmed[1]}, w={confirmed[2]}, h={confirmed[3]}")

        # ── STEP 3 ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="step-badge done">STEP 3 — Vibration Analysis</div>', unsafe_allow_html=True)

        if st.button("📊 Run Vibration Analysis", use_container_width=True):
            roi_tuple = st.session_state.get("confirmed_roi")
            if roi_tuple is None:
                st.warning("⚠️ Click **Confirm ROI** in Step 2 first.")
            else:
                an_status   = st.empty()
                an_progress = st.progress(0)
                an_status.info("🔄 Computing optical flow…")

                def _an_cb(cur, tot):
                    an_progress.progress(min(cur / tot, 1.0))

                try:
                    result = analyze_vibration(
                        input_path=mag_path, roi=roi_tuple,
                        fps=m_fps, method=of_method,
                        progress_callback=_an_cb,
                    )
                    an_progress.progress(1.0)
                    an_status.success("✅ Analysis complete!")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Dominant Freq", f"{result['dominant_hz']:.3f} Hz")
                    r2.metric("Period", f"{1/result['dominant_hz']:.3f} s" if result["dominant_hz"] > 0 else "—")
                    r3.metric("Amplitude", f"{result['dominant_amp']:.4f} px")
                    r4.metric("Frames analysed", str(len(result["motion"])))

                    fig = make_plot(result)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    csv_motion = io.StringIO()
                    csv_motion.write("time_s,motion_px_per_frame\n")
                    for t, m in zip(result["times"], result["motion"]):
                        csv_motion.write(f"{t:.6f},{m:.6f}\n")
                    csv_fft = io.StringIO()
                    csv_fft.write("freq_hz,power\n")
                    for f, p in zip(result["freqs"], result["power"]):
                        csv_fft.write(f"{f:.6f},{p:.6f}\n")

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button("⬇️ Motion Signal (CSV)", csv_motion.getvalue(),
                                           "motion_signal.csv", "text/csv", use_container_width=True)
                    with dl2:
                        st.download_button("⬇️ FFT Spectrum (CSV)", csv_fft.getvalue(),
                                           "fft_spectrum.csv", "text/csv", use_container_width=True)

                except Exception as e:
                    an_status.error(f"❌ Analysis error: {e}")
        else:
            if not st.session_state.get("confirmed_roi"):
                st.info("Set ROI coordinates above, click **Confirm ROI**, then click **Run Vibration Analysis**.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.7rem;color:#2d2d4e;padding:1rem 0;">
    Eulerian Video Magnification · MIT CSAIL Research · Built with Streamlit
</div>
""", unsafe_allow_html=True)
