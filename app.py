import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from magnify import process_video, analyze_vibration, ALPHA_CURVES

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

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #0a0a0f; color: #e8e8f0; }

[data-testid="stSidebar"] { background-color: #10101a !important; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

.main-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.8rem;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0; line-height: 1.1;
}
.sub-title {
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    color: #6b7280; margin-top: 4px; letter-spacing: 0.05em;
}
.info-card {
    background: #13131f; border: 1px solid #1e1e30;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.param-label {
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    color: #a78bfa; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 2px;
}
.section-header {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem;
    color: #38bdf8; border-bottom: 1px solid #1e1e30; padding-bottom: 6px; margin: 1.2rem 0 0.6rem;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    width: 100%; transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0284c7) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important; width: 100%;
}
[data-testid="stFileUploader"] {
    background: #13131f !important; border: 2px dashed #2d2d4e !important; border-radius: 12px !important;
}
.stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #38bdf8) !important; }
[data-testid="stMetric"] {
    background: #13131f; border: 1px solid #1e1e30; border-radius: 10px; padding: 1rem;
}
hr { border-color: #1e1e30 !important; }
.stSuccess, .stInfo { border-radius: 10px !important; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Motion Magnification Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Eulerian Video Magnification · Laplacian Pyramid · Temporal Filtering · Vibration Analysis</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Session state init ─────────────────────────────────────────────────────────
if "roi" not in st.session_state:
    st.session_state["roi"] = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Magnification Parameters")
    st.markdown("---")

    st.markdown('<div class="param-label">Alpha — Magnification Strength</div>', unsafe_allow_html=True)
    alpha = st.slider("Alpha", 10, 200, 100, 5)

    st.markdown('<div class="param-label">Lambda C — Spatial Wavelength (px)</div>', unsafe_allow_html=True)
    lambda_c = st.slider("Lambda C", 1, 100, 20, 1)

    st.markdown('<div class="param-label">fl — Low Frequency Cutoff (Hz)</div>', unsafe_allow_html=True)
    fl = st.slider("fl (Hz)", 0.1, 10.0, 1.0, 0.1)

    st.markdown('<div class="param-label">fh — High Frequency Cutoff (Hz)</div>', unsafe_allow_html=True)
    fh = st.slider("fh (Hz)", 1.0, 30.0, 14.0, 0.5)

    st.markdown('<div class="param-label">FPS — Video Frame Rate</div>', unsafe_allow_html=True)
    fps = st.slider("FPS", 15, 60, 30, 1)

    st.markdown("---")
    st.markdown("### 🔺 Pyramid Settings")

    st.markdown('<div class="param-label">Pyramid Levels (0 = default)</div>', unsafe_allow_html=True)
    n_levels_raw = st.slider("Pyramid Levels", 0, 8, 0, 1,
                             help="0 = pyrtools default. Fewer levels → coarser magnification. More → fine detail.")
    n_levels = None if n_levels_raw == 0 else n_levels_raw

    st.markdown('<div class="param-label">Alpha Curve — Per-Level Weighting</div>', unsafe_allow_html=True)
    alpha_curve = st.selectbox(
        "Alpha Curve",
        options=list(ALPHA_CURVES.keys()),
        index=0,
        help=(
            "auto – original spatial-freq attenuation (default)\n"
            "flat – same alpha at every level\n"
            "linear – tapers alpha from fine→coarse\n"
            "quadratic – strong emphasis on fine detail\n"
            "inverse – emphasises coarse/low-freq structure"
        ),
    )

    st.markdown("---")
    st.markdown("### 🌊 Vibration Analysis")

    of_method = st.selectbox(
        "Optical Flow Method",
        ["farneback", "lucas_kanade"],
        help="Farneback (dense, recommended) or Lucas-Kanade (sparse feature tracking).",
    )

    st.markdown("---")
    st.markdown("### 📖 Parameter Guide")
    st.markdown("""
<div style='font-size:0.78rem; color:#6b7280; line-height:1.8;'>
<b style='color:#a78bfa'>Alpha:</b> Start at 100. Heartbeat / breathing needs >80.<br>
<b style='color:#38bdf8'>fl / fh:</b> Heartbeat ≈ 1–2 Hz, breathing ≈ 0.5–1 Hz.<br>
<b style='color:#34d399'>Lambda C:</b> Lower → finer spatial detail. Start at 20.<br>
<b style='color:#f59e0b'>Pyr Levels:</b> 4–6 good for most videos. Fewer = faster.<br>
<b style='color:#f472b6'>Alpha Curve:</b> "quadratic" emphasises pixel-level micro-motion.
</div>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_plot(result: dict) -> plt.Figure:
    """Build a 2-panel matplotlib figure: motion signal + FFT spectrum."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5),
                                   facecolor="#0d0d18", constrained_layout=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d18")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e30")

    # — Motion signal
    ax1.plot(result["times"], result["motion"], color="#38bdf8", lw=1.2)
    ax1.set_xlabel("Time (s)", color="#9ca3af", fontsize=8)
    ax1.set_ylabel("Mean Flow (px/frame)", color="#9ca3af", fontsize=8)
    ax1.set_title("Optical-Flow Motion Signal (ROI)", color="#e8e8f0", fontsize=10, fontweight="bold")
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(color="#1e1e30", linestyle="--", linewidth=0.5, which="both")

    # — FFT power spectrum
    freqs = result["freqs"]
    power = result["power"]
    ax2.fill_between(freqs, power, alpha=0.35, color="#a78bfa")
    ax2.plot(freqs, power, color="#a78bfa", lw=1.2)
    dom = result["dominant_hz"]
    ax2.axvline(dom, color="#f59e0b", lw=1.5, linestyle="--", label=f"Peak: {dom:.3f} Hz")
    ax2.set_xlabel("Frequency (Hz)", color="#9ca3af", fontsize=8)
    ax2.set_ylabel("Power", color="#9ca3af", fontsize=8)
    ax2.set_title("FFT Vibration Spectrum", color="#e8e8f0", fontsize=10, fontweight="bold")
    ax2.legend(facecolor="#13131f", edgecolor="#1e1e30", labelcolor="#f59e0b", fontsize=8)
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.grid(color="#1e1e30", linestyle="--", linewidth=0.5, which="both")

    return fig


ROI_PICKER_HTML = """
<style>
  body {{ margin:0; background:#0d0d18; }}
  #canvas-wrap {{ position:relative; display:inline-block; }}
  canvas {{ display:block; cursor:crosshair; border:2px solid #1e1e30; border-radius:8px; }}
  #info {{ font-family:'Space Mono',monospace; font-size:0.72rem; color:#6b7280; margin:6px 0 4px; }}
  #roi-out {{ font-family:'Space Mono',monospace; font-size:0.78rem; color:#38bdf8;
              background:#13131f; border:1px solid #1e1e30; border-radius:6px;
              padding:6px 10px; margin-top:4px; min-height:22px; }}
  button {{
    margin-top:8px; background:linear-gradient(135deg,#7c3aed,#2563eb);
    color:#fff; border:none; border-radius:8px; font-family:'Syne',sans-serif;
    font-weight:600; font-size:0.9rem; padding:6px 22px; cursor:pointer;
  }}
  button:hover {{ opacity:0.85; }}
</style>
<div id="canvas-wrap">
  <canvas id="c" width="{CW}" height="{CH}"></canvas>
</div>
<div id="info">Click and drag to draw the ROI rectangle on the frame above.</div>
<div id="roi-out">No ROI selected yet.</div>
<button onclick="confirmROI()">✅ Confirm ROI</button>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const img = new Image();
img.onload = () => {{ ctx.drawImage(img, 0, 0, canvas.width, canvas.height); }};
img.src = 'data:image/jpeg;base64,{B64}';

// scale factors from display → actual video pixels
const scaleX = {VW} / canvas.width;
const scaleY = {VH} / canvas.height;

let drawing = false, startX=0, startY=0, rect={{x:0,y:0,w:0,h:0}};

canvas.addEventListener('mousedown', e => {{
  const r = canvas.getBoundingClientRect();
  startX = e.clientX - r.left;
  startY = e.clientY - r.top;
  drawing = true;
}});
canvas.addEventListener('mousemove', e => {{
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left;
  const cy = e.clientY - r.top;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2; ctx.setLineDash([5,3]);
  ctx.strokeRect(startX, startY, cx-startX, cy-startY);
  ctx.fillStyle='rgba(245,158,11,0.08)';
  ctx.fillRect(startX, startY, cx-startX, cy-startY);
}});
canvas.addEventListener('mouseup', e => {{
  if (!drawing) return;
  drawing = false;
  const r = canvas.getBoundingClientRect();
  const ex = e.clientX - r.left;
  const ey = e.clientY - r.top;
  rect = {{
    x: Math.round(Math.min(startX,ex)*scaleX),
    y: Math.round(Math.min(startY,ey)*scaleY),
    w: Math.round(Math.abs(ex-startX)*scaleX),
    h: Math.round(Math.abs(ey-startY)*scaleY)
  }};
  document.getElementById('roi-out').innerText =
    `ROI → x:${{rect.x}}  y:${{rect.y}}  w:${{rect.w}}  h:${{rect.h}}  (video pixels)`;
}});

function confirmROI() {{
  if (rect.w < 4 || rect.h < 4) {{
    alert('Please draw a larger ROI first.');
    return;
  }}
  window.parent.postMessage({{type:'roi', data: rect}}, '*');
  document.getElementById('roi-out').innerText =
    '✅ Sent to app — x:' + rect.x + '  y:' + rect.y + '  w:' + rect.w + '  h:' + rect.h;
}}
</script>
"""


# ── Main layout ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Drop your video here",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supports MP4, AVI, MOV, MKV",
    )

    tmp_input_path = None
    run_btn = False
    vid_w = vid_h = vid_fps = duration = total_frames = 0

    if uploaded_file:
        st.video(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_input_path = tmp.name

        cap = cv2.VideoCapture(tmp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        st.markdown("""
<div class="info-card">
<p style="color:#6b7280; font-size:0.9rem; margin:0;">
Upload a video to get started. Tune parameters in the sidebar, then click
<b style="color:#a78bfa">Run Motion Magnification</b>.
</p>
</div>
""", unsafe_allow_html=True)


with right_col:
    st.markdown("### 📥 Output")

    if uploaded_file and run_btn and tmp_input_path:
        output_path = tmp_input_path.replace(".mp4", "_magnified.mp4")

        status_box  = st.empty()
        progress_bar = st.progress(0)
        eta_box     = st.empty()
        status_box.info("⏳ Initialising magnification engine…")
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
                    unsafe_allow_html=True,
                )

        try:
            status_box.info("🔄 Processing frames…")
            out_path = process_video(
                input_path=tmp_input_path,
                output_path=output_path,
                alpha=alpha,
                lambda_c=lambda_c,
                fl=fl,
                fh=fh,
                fps=fps,
                progress_callback=progress_callback,
                n_levels=n_levels,
                alpha_curve=alpha_curve,
            )
            progress_bar.progress(1.0)
            eta_box.empty()
            elapsed_total = time.time() - start_time
            status_box.success(f"✅ Done! Processed in {elapsed_total:.1f}s")

            # H.264 output → plays inline in Streamlit
            with open(out_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Magnified Video",
                data=video_bytes,
                file_name="magnified_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

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


# ── Vibration Analysis Section ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌊 Vibration Analysis")

if not uploaded_file or tmp_input_path is None:
    st.info("Upload a video above to use the vibration analysis tools.")
else:
    import base64

    # Extract first frame for ROI picker
    cap_roi = cv2.VideoCapture(tmp_input_path)
    ret_roi, first_frame = cap_roi.read()
    cap_roi.release()

    if ret_roi:
        # Encode first frame as JPEG → base64 for the canvas picker
        CANVAS_W = 640
        CANVAS_H = int(vid_h * CANVAS_W / vid_w) if vid_w > 0 else 360
        _, buf = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_frame = base64.b64encode(buf.tobytes()).decode()

        st.markdown("### 🖼️ Draw ROI on First Frame")
        st.markdown(
            "Click and drag on the frame below to select the region of interest "
            "for vibration analysis, then click **Confirm ROI**."
        )

        # Render the interactive canvas picker inside an iframe
        html_code = ROI_PICKER_HTML.format(
            CW=CANVAS_W, CH=CANVAS_H,
            B64=b64_frame,
            VW=vid_w, VH=vid_h,
        )

        # We use a listener via st.components; the postMessage is caught by a
        # small JS bridge that writes into a hidden text_input via the DOM.
        # Simpler approach: use a form with manual coordinate entry as fallback.
        components.html(html_code, height=CANVAS_H + 120, scrolling=False)

        st.markdown("#### Or enter ROI coordinates manually")
        col_x, col_y, col_w, col_h = st.columns(4)
        with col_x:
            roi_x = st.number_input("X (px)", 0, vid_w - 1, 0, key="roi_x")
        with col_y:
            roi_y = st.number_input("Y (px)", 0, vid_h - 1, 0, key="roi_y")
        with col_w:
            roi_w = st.number_input("Width (px)", 4, vid_w, min(vid_w, 200), key="roi_w")
        with col_h:
            roi_h = st.number_input("Height (px)", 4, vid_h, min(vid_h, 200), key="roi_h")

        roi_tuple = (int(roi_x), int(roi_y), int(roi_w), int(roi_h))

        # Preview ROI on frame
        preview = first_frame.copy()
        cv2.rectangle(preview,
                      (roi_tuple[0], roi_tuple[1]),
                      (roi_tuple[0] + roi_tuple[2], roi_tuple[1] + roi_tuple[3]),
                      (245, 158, 11), 2)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, caption="ROI Preview (orange box)", use_container_width=True)

        run_analysis = st.button("📊 Run Vibration Analysis", use_container_width=True)

        if run_analysis:
            st.markdown("---")
            an_progress = st.progress(0)
            an_status   = st.empty()
            an_status.info("🔄 Computing optical flow…")

            def an_callback(cur, tot):
                an_progress.progress(cur / tot)

            try:
                result = analyze_vibration(
                    input_path=tmp_input_path,
                    roi=roi_tuple,
                    fps=vid_fps if vid_fps > 0 else fps,
                    method=of_method,
                    progress_callback=an_callback,
                )
                an_progress.progress(1.0)
                an_status.success("✅ Analysis complete!")

                # Metrics
                st.markdown("#### 📈 Vibration Results")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Dominant Freq", f"{result['dominant_hz']:.3f} Hz")
                r2.metric("Period", f"{1/result['dominant_hz']:.3f} s" if result["dominant_hz"] > 0 else "—")
                r3.metric("Amplitude", f"{result['dominant_amp']:.4f} px")
                r4.metric("Frames analysed", str(len(result["motion"])))

                # Plot
                fig = make_plot(result)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # Export CSV
                import io
                csv_buf = io.StringIO()
                csv_buf.write("time_s,motion_px_per_frame\n")
                for t, m in zip(result["times"], result["motion"]):
                    csv_buf.write(f"{t:.6f},{m:.6f}\n")
                st.download_button(
                    "⬇️ Download Motion Signal (CSV)",
                    data=csv_buf.getvalue(),
                    file_name="motion_signal.csv",
                    mime="text/csv",
                )

                csv_fft = io.StringIO()
                csv_fft.write("freq_hz,power\n")
                for f, p in zip(result["freqs"], result["power"]):
                    csv_fft.write(f"{f:.6f},{p:.6f}\n")
                st.download_button(
                    "⬇️ Download FFT Spectrum (CSV)",
                    data=csv_fft.getvalue(),
                    file_name="fft_spectrum.csv",
                    mime="text/csv",
                )

            except Exception as e:
                an_status.error(f"❌ Analysis error: {str(e)}")
    else:
        st.warning("Could not read first frame from video for ROI selection.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.7rem; color:#2d2d4e; padding:1rem 0;">
    Eulerian Video Magnification · MIT CSAIL Research · Built with Streamlit
</div>
""", unsafe_allow_html=True)
