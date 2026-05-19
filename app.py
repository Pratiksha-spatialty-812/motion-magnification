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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motion Magnification Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
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
.sub-title {
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
    color: #6b7280; margin-top: 4px; letter-spacing: 0.05em;
}
.step-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff; border-radius: 20px; padding: 3px 14px;
    letter-spacing: 0.06em; margin-bottom: 8px;
}
.step-badge.done   { background: linear-gradient(135deg, #059669, #0284c7); }
.step-badge.locked { background: #1e1e30; color: #4b5563; }
.info-card {
    background: #13131f; border: 1px solid #1e1e30;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.param-label {
    font-family: 'Space Mono', monospace; font-size: 0.73rem;
    color: #a78bfa; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 2px;
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
for key, default in [
    ("magnified_path",   None),
    ("original_bytes",   None),   # cached original video bytes
    ("mag_bytes",        None),   # cached magnified video bytes
    ("mag_vid_w",        0),
    ("mag_vid_h",        0),
    ("mag_vid_fps",      0.0),
    ("confirmed_roi",    None),   # (x,y,w,h) stored here — never wiped by widget reconciliation
    ("roi_input_val",    ""),     # bridge value written by JS
]:
    if key not in st.session_state:
        st.session_state[key] = default


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
    fps_sidebar = st.slider("FPS", 15, 60, 30, 1)
    st.markdown("---")
    st.markdown("### 🔺 Pyramid Settings")
    st.markdown('<div class="param-label">Pyramid Levels (0 = default)</div>', unsafe_allow_html=True)
    n_levels_raw = st.slider("Pyramid Levels", 0, 8, 0, 1)
    n_levels = None if n_levels_raw == 0 else n_levels_raw
    st.markdown('<div class="param-label">Alpha Curve</div>', unsafe_allow_html=True)
    alpha_curve = st.selectbox("Alpha Curve", list(ALPHA_CURVES.keys()), index=0)
    st.markdown("---")
    st.markdown("### 🌊 Vibration Analysis")
    of_method = st.selectbox("Optical Flow Method", ["farneback", "lucas_kanade"])
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.76rem; color:#6b7280; line-height:1.8;'>
<b style='color:#a78bfa'>Alpha:</b> Start at 100.<br>
<b style='color:#38bdf8'>fl/fh:</b> Heartbeat ≈ 1–2 Hz, breathing ≈ 0.5–1 Hz.<br>
<b style='color:#34d399'>Lambda C:</b> Lower → finer detail. Start at 20.<br>
<b style='color:#f59e0b'>Pyr Levels:</b> 4–6 works well for most videos.<br>
<b style='color:#f472b6'>Alpha Curve:</b> "quadratic" for micro-motion.
</div>
""", unsafe_allow_html=True)


# ── Helper: vibration plot ─────────────────────────────────────────────────────
def make_plot(result: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5),
                                   facecolor="#0d0d18", constrained_layout=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d18")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e30")
    ax1.plot(result["times"], result["motion"], color="#38bdf8", lw=1.2)
    ax1.set_xlabel("Time (s)", color="#9ca3af", fontsize=8)
    ax1.set_ylabel("Mean Flow (px/frame)", color="#9ca3af", fontsize=8)
    ax1.set_title("Optical-Flow Motion Signal — Magnified ROI", color="#e8e8f0", fontsize=10, fontweight="bold")
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(color="#1e1e30", linestyle="--", linewidth=0.5, which="both")
    freqs, power = result["freqs"], result["power"]
    ax2.fill_between(freqs, power, alpha=0.3, color="#a78bfa")
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


# ── Helper: ROI canvas HTML ────────────────────────────────────────────────────
def roi_canvas_html(b64_img: str, canvas_w: int, canvas_h: int,
                    vid_w: int, vid_h: int,
                    existing_roi: tuple | None) -> str:
    # Pre-populate box if ROI already confirmed
    preload_js = ""
    if existing_roi:
        rx, ry, rw, rh = existing_roi
        # Convert video coords → canvas coords
        preload_js = f"""
  window.addEventListener('load', function() {{
    var sx = {rx} * NW / VW;
    var sy = {ry} * NH / VH;
    var ex = ({rx} + {rw}) * NW / VW;
    var ey = ({ry} + {rh}) * NH / VH;
    rect = {{ dx:sx, dy:sy, dw:ex-sx, dh:ey-sy,
              x:{rx}, y:{ry}, w:{rw}, h:{rh} }};
    confirmed = true;
    ctx.drawImage(img, 0, 0, NW, NH);
    drawBox(rect.dx, rect.dy, rect.dw, rect.dh);
    document.getElementById('roi-label').innerText =
      'ROI → x:{rx}  y:{ry}  w:{rw}  h:{rh} px';
    var btn = document.getElementById('confirm-btn');
    btn.classList.add('sent');
    btn.innerText = '✅ ROI Confirmed — scroll down to run analysis';
  }});
"""

    return f"""
<style>
  body {{ margin:0; background:transparent; font-family:'Space Mono',monospace; overflow-x:hidden; }}
  canvas {{
    display:block;
    width:{canvas_w}px; height:{canvas_h}px;
    max-width:100%;
    cursor:crosshair; border-radius:10px; border:2px solid #1e1e30; box-sizing:border-box;
  }}
  #hint {{ font-size:0.72rem; color:#6b7280; margin:6px 0 8px; }}
  #roi-label {{
    font-size:0.78rem; color:#38bdf8; background:#13131f;
    border:1px solid #1e1e30; border-radius:6px; padding:5px 10px;
    display:inline-block; min-width:280px; margin-bottom:8px; word-break:break-all;
  }}
  #confirm-btn {{
    background:linear-gradient(135deg,#7c3aed,#2563eb); color:#fff;
    border:none; border-radius:8px; font-family:'Syne',sans-serif;
    font-weight:700; font-size:0.88rem; padding:7px 26px;
    cursor:pointer; transition:opacity .2s;
  }}
  #confirm-btn:hover {{ opacity:0.82; }}
  #confirm-btn.sent {{ background:linear-gradient(135deg,#059669,#0284c7); }}
</style>
<canvas id="c" width="{canvas_w}" height="{canvas_h}"></canvas>
<div id="hint">Click and drag to draw ROI, then click Confirm.</div>
<div id="roi-label">No ROI drawn yet</div><br>
<button id="confirm-btn" onclick="confirmROI()">✅ Confirm ROI</button>

<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const NW = {canvas_w};
const NH = {canvas_h};
const VW = {vid_w};
const VH = {vid_h};

const img = new Image();
img.onload = () => ctx.drawImage(img, 0, 0, NW, NH);
img.src    = 'data:image/jpeg;base64,{b64_img}';

function clientToCanvas(cx, cy) {{
  const r = canvas.getBoundingClientRect();
  return [(cx - r.left) * (NW / r.width), (cy - r.top) * (NH / r.height)];
}}

let drawing = false, sx = 0, sy = 0, rect = {{}}, confirmed = false;

function drawBox(x, y, w, h) {{
  ctx.save();
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2; ctx.setLineDash([6,3]);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = 'rgba(245,158,11,0.10)'; ctx.fillRect(x, y, w, h);
  ctx.restore();
}}
function redraw(cx, cy) {{
  ctx.clearRect(0, 0, NW, NH); ctx.drawImage(img, 0, 0, NW, NH);
  if (!drawing && confirmed) {{ drawBox(rect.dx, rect.dy, rect.dw, rect.dh); return; }}
  drawBox(sx, sy, cx - sx, cy - sy);
}}
function updateLabel(x, y, w, h) {{
  document.getElementById('roi-label').innerText =
    'ROI → x:' + x + '  y:' + y + '  w:' + w + '  h:' + h + ' px';
}}

canvas.addEventListener('mousedown', e => {{
  [sx, sy] = clientToCanvas(e.clientX, e.clientY);
  drawing = true; confirmed = false; e.preventDefault();
}});
canvas.addEventListener('mousemove', e => {{
  if (!drawing) return;
  const [cx, cy] = clientToCanvas(e.clientX, e.clientY);
  redraw(cx, cy);
  updateLabel(
    Math.round(Math.min(sx,cx)*VW/NW), Math.round(Math.min(sy,cy)*VH/NH),
    Math.round(Math.abs(cx-sx)*VW/NW), Math.round(Math.abs(cy-sy)*VH/NH)
  ); e.preventDefault();
}});
canvas.addEventListener('mouseup', e => {{
  if (!drawing) return; drawing = false;
  const [ex, ey] = clientToCanvas(e.clientX, e.clientY);
  rect = {{
    dx: Math.min(sx,ex), dy: Math.min(sy,ey),
    dw: Math.abs(ex-sx),  dh: Math.abs(ey-sy),
    x: Math.round(Math.min(sx,ex)*VW/NW), y: Math.round(Math.min(sy,ey)*VH/NH),
    w: Math.round(Math.abs(ex-sx)*VW/NW), h: Math.round(Math.abs(ey-sy)*VH/NH),
  }};
  confirmed = true; redraw(ex, ey); updateLabel(rect.x, rect.y, rect.w, rect.h);
  e.preventDefault();
}});
canvas.addEventListener('touchstart', e => {{
  const t = e.touches[0]; [sx, sy] = clientToCanvas(t.clientX, t.clientY);
  drawing = true; confirmed = false; e.preventDefault();
}}, {{passive:false}});
canvas.addEventListener('touchmove', e => {{
  if (!drawing) return;
  const t = e.touches[0]; const [cx,cy] = clientToCanvas(t.clientX, t.clientY);
  redraw(cx, cy); e.preventDefault();
}}, {{passive:false}});
canvas.addEventListener('touchend', e => {{
  if (!drawing) return; drawing = false;
  const t = e.changedTouches[0]; const [ex,ey] = clientToCanvas(t.clientX, t.clientY);
  rect = {{
    dx: Math.min(sx,ex), dy: Math.min(sy,ey),
    dw: Math.abs(ex-sx),  dh: Math.abs(ey-sy),
    x: Math.round(Math.min(sx,ex)*VW/NW), y: Math.round(Math.min(sy,ey)*VH/NH),
    w: Math.round(Math.abs(ex-sx)*VW/NW), h: Math.round(Math.abs(ey-sy)*VH/NH),
  }};
  confirmed = true; redraw(ex, ey); updateLabel(rect.x, rect.y, rect.w, rect.h);
  e.preventDefault();
}}, {{passive:false}});

// ── CONFIRM: post ROI to parent Streamlit via URL hash trick ─────────────────
function confirmROI() {{
  if (!confirmed || rect.w < 4 || rect.h < 4) {{ alert('Draw a larger ROI first.'); return; }}
  const val = rect.x + ',' + rect.y + ',' + rect.w + ',' + rect.h;
  // Method 1: postMessage to parent
  window.parent.postMessage({{ type: 'roi_confirmed', value: val }}, '*');
  // Method 2: also try the hidden input approach as fallback
  const inputs = window.parent.document.querySelectorAll('input[type="text"]');
  for (const inp of inputs) {{
    if (inp.getAttribute('aria-label') === '__roi_bridge__') {{
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value')
        .set.call(inp, val);
      inp.dispatchEvent(new Event('input', {{bubbles:true}}));
      break;
    }}
  }}
  const btn = document.getElementById('confirm-btn');
  btn.classList.add('sent');
  btn.innerText = '✅ ROI Confirmed — scroll down and click Run Vibration Analysis';
}}
{preload_js}
</script>
"""


# ── Helper: render video from bytes using base64 HTML5 player ─────────────────
def video_player_html(video_bytes: bytes, label: str = "") -> str:
    """
    Embed video as base64 data URI inside an HTML5 <video> tag.
    This bypasses Streamlit's st.video() codec/container issues entirely.
    """
    b64 = base64.b64encode(video_bytes).decode()
    return f"""
<div style="margin-bottom:8px">
  {f'<div style="font-size:0.85rem;color:#9ca3af;margin-bottom:4px;">{label}</div>' if label else ''}
  <video controls style="width:100%;border-radius:10px;border:1px solid #1e1e30;background:#000;">
    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
"""


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
    # Cache original bytes in session state so they survive reruns
    file_bytes = uploaded_file.read()
    if st.session_state["original_bytes"] is None or uploaded_file.name != st.session_state.get("uploaded_name"):
        st.session_state["original_bytes"] = file_bytes
        st.session_state["uploaded_name"]  = uploaded_file.name
        # New upload → reset downstream state
        st.session_state["magnified_path"] = None
        st.session_state["mag_bytes"]      = None
        st.session_state["confirmed_roi"]  = None
        st.session_state["roi_input_val"]  = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_input_path = tmp.name

    cap = cv2.VideoCapture(tmp_input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps      = cap.get(cv2.CAP_PROP_FPS)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / vid_fps if vid_fps > 0 else 0

    left_up, right_up = st.columns([1, 1], gap="large")

    with left_up:
        st.markdown("**Original**")
        # Render original via HTML5 player — always playable from cached bytes
        orig_bytes = st.session_state["original_bytes"]
        if orig_bytes:
            orig_b64 = base64.b64encode(orig_bytes).decode()
            st.markdown(f"""
<video controls style="width:100%;border-radius:10px;border:1px solid #1e1e30;background:#000;">
  <source src="data:video/mp4;base64,{orig_b64}" type="video/mp4">
</video>
""", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolution", f"{vid_w}×{vid_h}")
        m2.metric("FPS", f"{vid_fps:.0f}")
        m3.metric("Duration", f"{duration:.1f}s")

    with right_up:
        st.markdown("**Magnified output**")
        run_btn = st.button("🚀 Run Motion Magnification", use_container_width=True)

        if run_btn:
            st.session_state["magnified_path"] = None
            st.session_state["mag_bytes"]      = None
            st.session_state["confirmed_roi"]  = None
            st.session_state["roi_input_val"]  = ""

            output_path  = tmp_input_path.replace(".mp4", "_magnified.mp4")
            status_box   = st.empty()
            progress_bar = st.progress(0)
            eta_box      = st.empty()
            status_box.info("⏳ Initialising magnification engine…")
            t0 = time.time()

            def _mag_cb(cur, tot):
                pct = cur / tot
                progress_bar.progress(pct)
                elapsed = time.time() - t0
                if cur > 1 and pct > 0:
                    eta = (elapsed / pct) * (1 - pct)
                    eta_box.markdown(
                        f'<div style="font-family:Space Mono,monospace;font-size:0.75rem;'
                        f'color:#6b7280;">Frame {cur}/{tot} · ETA {eta:.0f}s</div>',
                        unsafe_allow_html=True,
                    )

            try:
                status_box.info("🔄 Processing frames…")
                out_path = process_video(
                    input_path=tmp_input_path,
                    output_path=output_path,
                    alpha=alpha, lambda_c=lambda_c,
                    fl=fl, fh=fh, fps=fps_sidebar,
                    progress_callback=_mag_cb,
                    n_levels=n_levels, alpha_curve=alpha_curve,
                )
                progress_bar.progress(1.0)
                eta_box.empty()
                status_box.success(f"✅ Done in {time.time()-t0:.1f}s — scroll down to analyse")

                with open(out_path, "rb") as vf:
                    st.session_state["mag_bytes"] = vf.read()
                st.session_state["magnified_path"] = out_path
                st.session_state["mag_vid_w"]   = vid_w
                st.session_state["mag_vid_h"]   = vid_h
                st.session_state["mag_vid_fps"] = vid_fps if vid_fps > 0 else float(fps_sidebar)

            except Exception as e:
                status_box.error(f"❌ Error: {e}")
                progress_bar.empty(); eta_box.empty()

        # Always render magnified video from cached bytes (survives all reruns)
        mag_bytes = st.session_state.get("mag_bytes")
        if mag_bytes:
            mag_b64 = base64.b64encode(mag_bytes).decode()
            st.markdown(f"""
<video controls style="width:100%;border-radius:10px;border:1px solid #1e1e30;background:#000;">
  <source src="data:video/mp4;base64,{mag_b64}" type="video/mp4">
</video>
""", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Magnified Video",
                data=mag_bytes,
                file_name="magnified_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

else:
    # Clear state if file removed
    st.session_state["original_bytes"]  = None
    st.session_state["magnified_path"]  = None
    st.session_state["mag_bytes"]       = None
    st.session_state["confirmed_roi"]   = None
    st.session_state["roi_input_val"]   = ""
    st.markdown("""
<div class="info-card">
<p style="color:#6b7280; font-size:0.9rem; margin:0;">
Upload a video above, tune parameters in the sidebar, then click
<b style="color:#a78bfa">Run Motion Magnification</b>.
</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ROI  |  STEP 3 — Vibration Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

mag_path  = st.session_state.get("magnified_path")
mag_ready = bool(mag_path and os.path.exists(mag_path))

if not mag_ready:
    st.markdown('<div class="step-badge locked">STEP 2 — Draw ROI &nbsp;·&nbsp; run magnification first</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-badge locked" style="margin-top:6px">STEP 3 — Vibration Analysis &nbsp;·&nbsp; run magnification first</div>', unsafe_allow_html=True)

else:
    # ── STEP 2 ────────────────────────────────────────────────────────────────
    st.markdown('<div class="step-badge done">STEP 2 — Draw ROI on Magnified Video</div>', unsafe_allow_html=True)
    st.markdown("Draw a bounding box on the magnified frame. Vibration analysis will run inside this region.")

    m_w   = st.session_state["mag_vid_w"]
    m_h   = st.session_state["mag_vid_h"]
    m_fps = st.session_state["mag_vid_fps"]

    cap_m = cv2.VideoCapture(mag_path)
    ret_m, first_mag_frame = cap_m.read()
    cap_m.release()

    if not ret_m:
        st.warning("Could not read first frame of magnified video.")
    else:
        CANVAS_W = min(640, m_w) if m_w > 0 else 640
        CANVAS_H = int(m_h * CANVAS_W / m_w) if m_w > 0 else 360

        _, buf = cv2.imencode(".jpg", first_mag_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        b64_frame = base64.b64encode(buf.tobytes()).decode()

        # ── ROI bridge: hidden text input that JS writes into ─────────────────
        # We use on_change to immediately move the value into confirmed_roi
        def _on_roi_change():
            raw = st.session_state.get("roi_input_val", "").strip()
            if raw:
                try:
                    parts = [int(v) for v in raw.split(",")]
                    if len(parts) == 4 and parts[2] >= 4 and parts[3] >= 4:
                        st.session_state["confirmed_roi"] = tuple(parts)
                except ValueError:
                    pass

        roi_input = st.text_input(
            label="__roi_bridge__",
            key="roi_input_val",
            label_visibility="collapsed",
            on_change=_on_roi_change,
        )
        # Also parse on every render (catches value already in state)
        _on_roi_change()

        components.html(
            roi_canvas_html(b64_frame, CANVAS_W, CANVAS_H, m_w, m_h,
                            existing_roi=st.session_state.get("confirmed_roi")),
            height=CANVAS_H + 160,
            scrolling=False,
        )

        roi_tuple = st.session_state.get("confirmed_roi")

        if roi_tuple:
            preview = first_mag_frame.copy()
            cv2.rectangle(
                preview,
                (roi_tuple[0], roi_tuple[1]),
                (roi_tuple[0] + roi_tuple[2], roi_tuple[1] + roi_tuple[3]),
                (245, 158, 11), 2,
            )
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption=f"Confirmed ROI — x:{roi_tuple[0]}  y:{roi_tuple[1]}  w:{roi_tuple[2]}  h:{roi_tuple[3]}",
                use_container_width=True,
            )

        # ── STEP 3 ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="step-badge done">STEP 3 — Vibration Analysis</div>', unsafe_allow_html=True)

        run_analysis = st.button("📊 Run Vibration Analysis", use_container_width=True)

        if run_analysis:
            roi_tuple = st.session_state.get("confirmed_roi")
            if roi_tuple is None:
                st.warning("⚠️ Draw and confirm an ROI in Step 2 first.")
            else:
                an_status   = st.empty()
                an_progress = st.progress(0)
                an_status.info("🔄 Computing optical flow on magnified video…")

                def _an_cb(cur, tot):
                    an_progress.progress(min(cur / tot, 1.0))

                try:
                    result = analyze_vibration(
                        input_path=mag_path,
                        roi=roi_tuple,
                        fps=m_fps,
                        method=of_method,
                        progress_callback=_an_cb,
                    )
                    an_progress.progress(1.0)
                    an_status.success("✅ Analysis complete!")

                    st.markdown("#### 📈 Vibration Results")
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
                        st.download_button("⬇️ Motion Signal (CSV)", data=csv_motion.getvalue(),
                                           file_name="motion_signal.csv", mime="text/csv", use_container_width=True)
                    with dl2:
                        st.download_button("⬇️ FFT Spectrum (CSV)", data=csv_fft.getvalue(),
                                           file_name="fft_spectrum.csv", mime="text/csv", use_container_width=True)

                except Exception as e:
                    an_status.error(f"❌ Analysis error: {e}")

        elif not st.session_state.get("confirmed_roi"):
            st.info("Confirm an ROI in Step 2 above, then click **Run Vibration Analysis**.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:0.7rem; color:#2d2d4e; padding:1rem 0;">
    Eulerian Video Magnification · MIT CSAIL Research · Built with Streamlit
</div>
""", unsafe_allow_html=True)
