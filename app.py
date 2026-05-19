import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import os
import time
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
st.markdown(
    "Draw a bounding box on the first frame to define your region of interest. "
    "Optical flow is computed inside that region across every frame of the **original video**, "
    "then FFT reveals the dominant vibration frequency."
)

if not uploaded_file or tmp_input_path is None:
    st.info("Upload a video above to enable vibration analysis.")
else:
    import base64, io

    # ── Extract first frame ────────────────────────────────────────────────────
    cap_roi = cv2.VideoCapture(tmp_input_path)
    ret_roi, first_frame = cap_roi.read()
    cap_roi.release()

    if not ret_roi:
        st.warning("Could not read first frame for ROI selection.")
    else:
        CANVAS_W = min(720, vid_w)
        CANVAS_H = int(vid_h * CANVAS_W / vid_w) if vid_w > 0 else 360
        _, buf = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        b64_frame = base64.b64encode(buf.tobytes()).decode()

        # ── ROI state stored in session_state["roi_coords"] as "x,y,w,h" ──────
        if "roi_coords" not in st.session_state:
            st.session_state["roi_coords"] = ""

        # Hidden text_input that JS writes into via DOM manipulation
        # We render it visually hidden with CSS
        st.markdown("""
<style>
div[data-testid="stTextInput"]:has(input[aria-label="__roi_hidden__"]) {
    position: absolute; opacity: 0; pointer-events: none; height: 0; overflow: hidden;
}
</style>""", unsafe_allow_html=True)

        roi_hidden = st.text_input("__roi_hidden__", key="roi_coords", label_visibility="hidden")

        # ── Canvas ROI picker ──────────────────────────────────────────────────
        CANVAS_HTML = f"""
<style>
  body {{ margin:0; background:transparent; font-family:'Space Mono',monospace; }}
  #wrap {{ position:relative; display:inline-block; width:100%; }}
  canvas {{ display:block; cursor:crosshair; border-radius:10px;
            border:2px solid #1e1e30; width:100%; height:auto; }}
  #hint  {{ font-size:0.72rem; color:#6b7280; margin: 6px 0 8px; }}
  #roi-label {{ font-size:0.78rem; color:#38bdf8; background:#13131f;
                border:1px solid #1e1e30; border-radius:6px;
                padding:5px 10px; display:inline-block; min-width:260px;
                margin-bottom:8px; }}
  #confirm-btn {{
    background:linear-gradient(135deg,#7c3aed,#2563eb);
    color:#fff; border:none; border-radius:8px;
    font-family:'Syne',sans-serif; font-weight:700; font-size:0.88rem;
    padding:7px 26px; cursor:pointer; transition:opacity .2s;
  }}
  #confirm-btn:hover {{ opacity:0.82; }}
  #confirm-btn.sent {{ background:linear-gradient(135deg,#059669,#0284c7); }}
</style>
<div id="wrap">
  <canvas id="c" width="{CANVAS_W}" height="{CANVAS_H}"></canvas>
</div>
<div id="hint">Click and drag to draw ROI, then click Confirm.</div>
<div id="roi-label">No ROI drawn yet</div><br>
<button id="confirm-btn" onclick="confirmROI()">✅ Confirm ROI</button>

<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const img    = new Image();
img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
img.src    = 'data:image/jpeg;base64,{b64_frame}';

const scaleX = {vid_w} / {CANVAS_W};
const scaleY = {vid_h} / {CANVAS_H};

let drawing=false, sx=0, sy=0, rect={{x:0,y:0,w:0,h:0}}, confirmed=false;

function redraw(cx, cy) {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
  if (!drawing && confirmed) {{
    // draw confirmed box
    drawBox(rect.dx, rect.dy, rect.dw, rect.dh);
    return;
  }}
  const dw = cx - sx, dh = cy - sy;
  drawBox(sx, sy, dw, dh);
}}

function drawBox(x,y,w,h) {{
  ctx.save();
  ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2; ctx.setLineDash([6,3]);
  ctx.strokeRect(x,y,w,h);
  ctx.fillStyle='rgba(245,158,11,0.10)';
  ctx.fillRect(x,y,w,h);
  ctx.restore();
}}

canvas.addEventListener('mousedown', e => {{
  const r=canvas.getBoundingClientRect();
  sx = (e.clientX-r.left)*(canvas.width/r.width);
  sy = (e.clientY-r.top)*(canvas.height/r.height);
  drawing=true; confirmed=false;
}});
canvas.addEventListener('mousemove', e => {{
  if (!drawing) return;
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*(canvas.width/r.width);
  const cy=(e.clientY-r.top)*(canvas.height/r.height);
  redraw(cx,cy);
  const vx=Math.round(Math.min(sx,cx)*scaleX);
  const vy=Math.round(Math.min(sy,cy)*scaleY);
  const vw=Math.round(Math.abs(cx-sx)*scaleX);
  const vh=Math.round(Math.abs(cy-sy)*scaleY);
  document.getElementById('roi-label').innerText =
    'ROI → x:'+vx+' y:'+vy+' w:'+vw+' h:'+vh+' px';
}});
canvas.addEventListener('mouseup', e => {{
  if (!drawing) return;
  drawing=false;
  const r=canvas.getBoundingClientRect();
  const ex=(e.clientX-r.left)*(canvas.width/r.width);
  const ey=(e.clientY-r.top)*(canvas.height/r.height);
  rect = {{
    dx:Math.min(sx,ex), dy:Math.min(sy,ey),
    dw:Math.abs(ex-sx),  dh:Math.abs(ey-sy),
    x:Math.round(Math.min(sx,ex)*scaleX),
    y:Math.round(Math.min(sy,ey)*scaleY),
    w:Math.round(Math.abs(ex-sx)*scaleX),
    h:Math.round(Math.abs(ey-sy)*scaleY),
  }};
  confirmed=true;
  redraw(ex,ey);
}});

function confirmROI() {{
  if (!confirmed || rect.w < 4 || rect.h < 4) {{
    alert('Please draw a larger ROI first.'); return;
  }}
  const val = rect.x+','+rect.y+','+rect.w+','+rect.h;
  // Write into the hidden Streamlit text_input and fire React change event
  const inputs = window.parent.document.querySelectorAll('input[type="text"]');
  for (const inp of inputs) {{
    if (inp.getAttribute('aria-label') === '__roi_hidden__' ||
        inp.closest('[data-testid="stTextInput"]')) {{
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value').set;
      nativeInputValueSetter.call(inp, val);
      inp.dispatchEvent(new Event('input', {{bubbles:true}}));
      break;
    }}
  }}
  const btn = document.getElementById('confirm-btn');
  btn.classList.add('sent');
  btn.innerText = '✅ ROI Confirmed — click Run Analysis';
}}
</script>
"""
        components.html(CANVAS_HTML, height=CANVAS_H + 110, scrolling=False)

        # ── Parse confirmed ROI ────────────────────────────────────────────────
        roi_tuple = None
        raw = st.session_state.get("roi_coords", "").strip()
        if raw:
            try:
                parts = [int(v) for v in raw.split(",")]
                if len(parts) == 4 and parts[2] >= 4 and parts[3] >= 4:
                    roi_tuple = tuple(parts)
            except ValueError:
                pass

        if roi_tuple:
            # Show a static preview with the confirmed box
            preview = first_frame.copy()
            cv2.rectangle(
                preview,
                (roi_tuple[0], roi_tuple[1]),
                (roi_tuple[0] + roi_tuple[2], roi_tuple[1] + roi_tuple[3]),
                (245, 158, 11), 2,
            )
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption=f"Confirmed ROI  x:{roi_tuple[0]} y:{roi_tuple[1]} "
                        f"w:{roi_tuple[2]} h:{roi_tuple[3]}",
                use_container_width=True,
            )

            run_analysis = st.button("📊 Run Vibration Analysis", use_container_width=True)

            if run_analysis:
                an_status   = st.empty()
                an_progress = st.progress(0)
                an_status.info("🔄 Computing optical flow on original video…")

                def an_callback(cur, tot):
                    an_progress.progress(min(cur / tot, 1.0))

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

                    st.markdown("#### 📈 Vibration Results")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Dominant Freq", f"{result['dominant_hz']:.3f} Hz")
                    r2.metric(
                        "Period",
                        f"{1/result['dominant_hz']:.3f} s" if result["dominant_hz"] > 0 else "—",
                    )
                    r3.metric("Amplitude", f"{result['dominant_amp']:.4f} px")
                    r4.metric("Frames analysed", str(len(result["motion"])))

                    fig = make_plot(result)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # CSV downloads
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
                        st.download_button(
                            "⬇️ Motion Signal (CSV)",
                            data=csv_motion.getvalue(),
                            file_name="motion_signal.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with dl2:
                        st.download_button(
                            "⬇️ FFT Spectrum (CSV)",
                            data=csv_fft.getvalue(),
                            file_name="fft_spectrum.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                except Exception as e:
                    an_status.error(f"❌ Analysis error: {str(e)}")
        else:
            st.info("Draw and confirm a ROI on the frame above, then click **Run Vibration Analysis**.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.7rem; color:#2d2d4e; padding:1rem 0;">
    Eulerian Video Magnification · MIT CSAIL Research · Built with Streamlit
</div>
""", unsafe_allow_html=True)
