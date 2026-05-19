import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import os
import io
import time
import base64
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
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
.param-label {
    font-family: 'Space Mono', monospace; font-size: 0.73rem;
    color: #a78bfa; text-transform: uppercase; margin-bottom: 2px;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important; width: 100%;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0284c7) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important; width: 100%;
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
    ("orig_tmp_path",    None),
    ("orig_vid_bytes",   None),
    ("magnified_path",   None),
    ("mag_vid_bytes",    None),
    ("mag_vid_w",        0),
    ("mag_vid_h",        0),
    ("mag_vid_fps",      0.0),
    ("confirmed_roi",    None),
    ("pending_roi",      None),   # roi drawn but not yet confirmed
    ("last_upload_name", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_browser_mp4_bytes(input_path: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
            "-an",
            out_path,
        ], capture_output=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(out_path):
            st.error(f"ffmpeg failed: {r.stderr.decode()}")
            return b""
        with open(out_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.error("ffmpeg not found. Add `ffmpeg` to packages.txt in your repo root.")
        return b""
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def make_plot(result: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5),
                                   facecolor="#0d0d18", constrained_layout=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d18")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e30")
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
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
    n_levels_raw = st.slider("Pyramid Levels (0=auto)", 0, 8, 0, 1)
    n_levels = None if n_levels_raw == 0 else n_levels_raw
    alpha_curve = st.selectbox("Alpha Curve", list(ALPHA_CURVES.keys()), index=0)
    st.markdown("---")
    of_method = st.selectbox("Optical Flow Method", ["farneback", "lucas_kanade"])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">STEP 1 — Upload &amp; Magnify</div>', unsafe_allow_html=True)
st.markdown("### 📤 Upload Video")

uploaded_file = st.file_uploader("Drop your video here", type=["mp4", "avi", "mov", "mkv"])

tmp_input_path = None
vid_w = vid_h = total_frames = 0
vid_fps = 0.0

if uploaded_file:
    fname = uploaded_file.name

    if st.session_state["last_upload_name"] != fname:
        raw_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as tmp:
            tmp.write(raw_bytes)
            st.session_state["orig_tmp_path"] = tmp.name
        st.session_state["last_upload_name"] = fname
        st.session_state["orig_vid_bytes"]   = None
        st.session_state["magnified_path"]   = None
        st.session_state["mag_vid_bytes"]    = None
        st.session_state["confirmed_roi"]    = None
        st.session_state["pending_roi"]      = None

    tmp_input_path = st.session_state["orig_tmp_path"]

    if st.session_state["orig_vid_bytes"] is None:
        with st.spinner("Preparing original video for playback…"):
            b = to_browser_mp4_bytes(tmp_input_path)
            st.session_state["orig_vid_bytes"] = b

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
        orig_bytes = st.session_state["orig_vid_bytes"]
        if orig_bytes:
            st.video(orig_bytes, format="video/mp4")
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolution", f"{vid_w}×{vid_h}")
        m2.metric("FPS", f"{vid_fps:.0f}")
        m3.metric("Duration", f"{duration:.1f}s")

    with right_up:
        st.markdown("**Magnified output**")
        run_btn = st.button("🚀 Run Motion Magnification", use_container_width=True)

        if run_btn:
            st.session_state["magnified_path"] = None
            st.session_state["mag_vid_bytes"]  = None
            st.session_state["confirmed_roi"]  = None
            st.session_state["pending_roi"]    = None

            raw_out  = tmp_input_path.replace(os.path.splitext(tmp_input_path)[1], "_mag_raw.mp4")

            status   = st.empty()
            prog     = st.progress(0)
            eta_slot = st.empty()
            t0 = time.time()

            def _cb(cur, tot):
                prog.progress(cur / tot)
                el = time.time() - t0
                if cur > 1:
                    eta_slot.caption(f"Frame {cur}/{tot} · ETA {(el/(cur/tot))*(1-cur/tot):.0f}s")

            try:
                status.info("🔄 Processing frames…")
                out_path = process_video(
                    input_path=tmp_input_path, output_path=raw_out,
                    alpha=alpha, lambda_c=lambda_c, fl=fl, fh=fh, fps=fps_sidebar,
                    progress_callback=_cb, n_levels=n_levels, alpha_curve=alpha_curve,
                )
                prog.progress(1.0); eta_slot.empty()

                status.info("🎞 Encoding for browser playback…")
                mag_bytes = to_browser_mp4_bytes(out_path)
                if not mag_bytes:
                    with open(out_path, "rb") as f:
                        mag_bytes = f.read()

                st.session_state["mag_vid_bytes"]  = mag_bytes
                st.session_state["magnified_path"] = out_path
                st.session_state["mag_vid_w"]      = vid_w
                st.session_state["mag_vid_h"]      = vid_h
                st.session_state["mag_vid_fps"]    = vid_fps if vid_fps > 0 else float(fps_sidebar)
                status.success(f"✅ Done in {time.time()-t0:.1f}s")

            except Exception as e:
                status.error(f"❌ {e}")
                prog.empty(); eta_slot.empty()

        mag_bytes = st.session_state.get("mag_vid_bytes")
        if mag_bytes:
            st.video(mag_bytes, format="video/mp4")
            st.download_button(
                "⬇️ Download Magnified Video", data=mag_bytes,
                file_name="magnified_output.mp4", mime="video/mp4",
                use_container_width=True,
            )

else:
    for k in ["orig_tmp_path", "orig_vid_bytes", "magnified_path",
              "mag_vid_bytes", "confirmed_roi", "pending_roi", "last_upload_name"]:
        st.session_state[k] = None
    st.info("Upload a video above, tune parameters in the sidebar, then click **Run Motion Magnification**.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ROI via streamlit-drawable-canvas
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

mag_path  = st.session_state.get("magnified_path")
mag_ready = bool(mag_path and os.path.exists(mag_path))

if not mag_ready:
    st.markdown('<div class="step-badge locked">STEP 2 — Draw ROI · run magnification first</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-badge locked" style="margin-top:6px">STEP 3 — Vibration Analysis · run magnification first</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="step-badge done">STEP 2 — Draw ROI on Magnified Frame</div>', unsafe_allow_html=True)
    st.markdown("Draw a rectangle on the frame below, then click **✅ Confirm ROI**.")

    m_w   = st.session_state["mag_vid_w"]
    m_h   = st.session_state["mag_vid_h"]
    m_fps = st.session_state["mag_vid_fps"]

    # Extract first frame
    cap_m = cv2.VideoCapture(mag_path)
    ret_m, first_frame = cap_m.read()
    cap_m.release()

    if not ret_m:
        st.warning("Could not read first frame of magnified video.")
    else:
        # Canvas sizing — max 700px wide, maintain aspect ratio
        CANVAS_W = min(700, m_w) if m_w > 0 else 700
        CANVAS_H = int(m_h * CANVAS_W / m_w) if m_w > 0 else 400

        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb).resize((CANVAS_W, CANVAS_H))

        # ── Drawable canvas — coordinates come back directly to Python ────────
        canvas_result = st_canvas(
            fill_color   = "rgba(245, 158, 11, 0.15)",
            stroke_width = 2,
            stroke_color = "#f59e0b",
            background_image  = pil_img,
            update_streamlit  = True,
            width        = CANVAS_W,
            height       = CANVAS_H,
            drawing_mode = "rect",
            key          = "roi_canvas",
        )

        # ── Read coordinates from canvas result into pending_roi ──────────────
        pending_roi = None
        if (
            canvas_result.json_data is not None
            and len(canvas_result.json_data.get("objects", [])) > 0
        ):
            obj = canvas_result.json_data["objects"][-1]   # last drawn rect
            scale_x = m_w / CANVAS_W
            scale_y = m_h / CANVAS_H
            rx = int(obj["left"]   * scale_x)
            ry = int(obj["top"]    * scale_y)
            rw = int(obj["width"]  * scale_x)
            rh = int(obj["height"] * scale_y)

            if rw >= 4 and rh >= 4:
                pending_roi = (rx, ry, rw, rh)
                # Show live preview of what will be confirmed
                st.caption(f"📐 Drawn — x:{rx}  y:{ry}  w:{rw}  h:{rh}  ← click Confirm ROI to lock")

        # ── Confirm / Clear buttons ───────────────────────────────────────────
        col_confirm, col_clear = st.columns([3, 1])
        with col_confirm:
            if st.button("✅ Confirm ROI", use_container_width=True):
                if pending_roi:
                    st.session_state["confirmed_roi"] = pending_roi
                    st.success(
                        f"✅ ROI confirmed — x:{pending_roi[0]}  y:{pending_roi[1]}"
                        f"  w:{pending_roi[2]}  h:{pending_roi[3]}"
                    )
                else:
                    st.warning("Draw a rectangle on the canvas above first.")
        with col_clear:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state["confirmed_roi"] = None

        # ── Overlay preview of confirmed ROI on frame ─────────────────────────
        confirmed = st.session_state.get("confirmed_roi")
        if confirmed:
            rx, ry, rw, rh = confirmed
            preview = first_frame.copy()
            overlay = preview.copy()
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (245, 158, 11), -1)
            preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)
            cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (245, 158, 11), 2)
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption=f"Active ROI — x:{rx}  y:{ry}  w:{rw}  h:{rh}",
                use_container_width=True,
            )

        # ══════════════════════════════════════════════════════════════════════
        #  STEP 3 — Vibration Analysis
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="step-badge done">STEP 3 — Vibration Analysis</div>', unsafe_allow_html=True)

        if st.button("📊 Run Vibration Analysis", use_container_width=True):
            roi_tuple = st.session_state.get("confirmed_roi")
            if roi_tuple is None:
                st.warning("⚠️ Draw a rectangle and click **Confirm ROI** in Step 2 first.")
            else:
                an_status = st.empty()
                an_prog   = st.progress(0)
                an_status.info("🔄 Computing optical flow on magnified video…")

                def _an_cb(cur, tot):
                    an_prog.progress(min(cur / tot, 1.0))

                try:
                    result = analyze_vibration(
                        input_path=mag_path, roi=roi_tuple,
                        fps=m_fps, method=of_method,
                        progress_callback=_an_cb,
                    )
                    an_prog.progress(1.0)
                    an_status.success("✅ Analysis complete!")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Dominant Freq", f"{result['dominant_hz']:.3f} Hz")
                    r2.metric("Period",
                              f"{1/result['dominant_hz']:.3f} s"
                              if result["dominant_hz"] > 0 else "—")
                    r3.metric("Amplitude", f"{result['dominant_amp']:.4f} px")
                    r4.metric("Frames",    str(len(result["motion"])))

                    fig = make_plot(result)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    csv_m = io.StringIO()
                    csv_m.write("time_s,motion_px_per_frame\n")
                    for t, m in zip(result["times"], result["motion"]):
                        csv_m.write(f"{t:.6f},{m:.6f}\n")
                    csv_f = io.StringIO()
                    csv_f.write("freq_hz,power\n")
                    for f, p in zip(result["freqs"], result["power"]):
                        csv_f.write(f"{f:.6f},{p:.6f}\n")

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button("⬇️ Motion Signal (CSV)", csv_m.getvalue(),
                                           "motion_signal.csv", "text/csv",
                                           use_container_width=True)
                    with dl2:
                        st.download_button("⬇️ FFT Spectrum (CSV)", csv_f.getvalue(),
                                           "fft_spectrum.csv", "text/csv",
                                           use_container_width=True)

                except Exception as e:
                    an_status.error(f"❌ {e}")
        else:
            if not st.session_state.get("confirmed_roi"):
                st.info("Draw and confirm an ROI above, then click **Run Vibration Analysis**.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.7rem;color:#2d2d4e;padding:1rem 0;">
    Eulerian Video Magnification · MIT CSAIL Research · Built with Streamlit
</div>
""", unsafe_allow_html=True)
