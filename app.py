import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import os
import io
import time
import base64
import subprocess
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from magnify import process_video, analyze_vibration, ALPHA_CURVES

# ── Static ffmpeg bootstrap ────────────────────────────────────────────────────
_FFMPEG_DIR = "/tmp/ffmpeg_bin"

def _ensure_ffmpeg():
    """
    Ensure ffmpeg/ffprobe are available.
    - If already on PATH (system install), add nothing but verify.
    - Otherwise download a static GPL build into /tmp/ffmpeg_bin and
      prepend that dir to PATH so all subprocess calls find it.
    """
    # Check if a working ffmpeg is already on PATH
    r = subprocess.run(["which", "ffmpeg"], capture_output=True)
    if r.returncode == 0:
        return  # system ffmpeg is fine

    # Already downloaded in a previous Streamlit rerun?
    if os.path.isfile(os.path.join(_FFMPEG_DIR, "ffmpeg")):
        _prepend_path()
        return

    ffmpeg_url = (
        "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linux64-gpl.tar.xz"
    )
    import urllib.request, tarfile as _tarfile
    os.makedirs(_FFMPEG_DIR, exist_ok=True)
    local = "/tmp/ffmpeg.tar.xz"
    urllib.request.urlretrieve(ffmpeg_url, local)
    with _tarfile.open(local) as t:
        for m in t.getmembers():
            if m.name.endswith("/ffmpeg") or m.name.endswith("/ffprobe"):
                m.name = os.path.basename(m.name)   # strip directory prefix
                t.extract(m, _FFMPEG_DIR)
    os.chmod(os.path.join(_FFMPEG_DIR, "ffmpeg"),  0o755)
    os.chmod(os.path.join(_FFMPEG_DIR, "ffprobe"), 0o755)
    try:
        os.unlink(local)
    except OSError:
        pass
    _prepend_path()

def _prepend_path():
    """Add our static binary dir to the front of PATH for this process."""
    current = os.environ.get("PATH", "")
    if _FFMPEG_DIR not in current:
        os.environ["PATH"] = _FFMPEG_DIR + ":" + current

_ensure_ffmpeg()

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
#MainMenu { display: none !important; }
footer { display: none !important; }
header { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="manage-app-button"] { display: none !important; }
section[data-testid="stBottom"] { display: none !important; }
.stDeployButton { display: none !important; }
</style>
<script>
function removeElements() {
    const selectors = [
        '[data-testid="manage-app-button"]',
        '[data-testid="stBottom"]',
        '[data-testid="stToolbar"]',
        'footer', 'header'
    ];
    selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => el.remove());
    });
}
removeElements();
setTimeout(removeElements, 500);
setTimeout(removeElements, 2000);
</script>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 Motion Magnification </div>', unsafe_allow_html=True)
st.markdown("---")

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("orig_tmp_path",    None),
    ("orig_preview_path", None),
    ("magnified_path",   None),
    ("mag_vid_bytes",    None),
    ("mag_vid_w",        0),
    ("mag_vid_h",        0),
    ("mag_vid_fps",      0.0),
    ("confirmed_roi",    None),
    ("pending_roi",      None),
    ("last_upload_name", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_ffmpeg_capabilities():
    caps = {"hevc": False, "h264": False, "libx264": False}
    try:
        r = subprocess.run(
            ["ffmpeg", "-codecs"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10
        )
        out = r.stdout.decode(errors="replace")
        if "hevc" in out or "libx265" in out:
            caps["hevc"] = True
        if "h264" in out or "libx264" in out:
            caps["h264"] = True
        if "libx264" in out:
            caps["libx264"] = True
    except Exception:
        pass
    return caps


def _get_video_info(input_path: str) -> dict:
    info = {
        "codec": "unknown",
        "rotate_tag": 0,
        "display_matrix_rotation": 0,
        "width": 0,
        "height": 0,
        "duration": 0.0,
    }
    try:
        stderr_tmp = tempfile.mktemp(suffix=".txt")
        with open(stderr_tmp, "wb") as ef:
            r = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,duration"
                    ":stream_tags=rotate"
                    ":stream_side_data=rotation",
                    "-of", "default=noprint_wrappers=1",
                    input_path,
                ],
                stdout=subprocess.PIPE,
                stderr=ef,
                timeout=20,
            )
        for line in r.stdout.decode(errors="replace").splitlines():
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip()
            if k == "codec_name":
                info["codec"] = v
            elif k == "width" and v.lstrip("-").isdigit():
                info["width"] = int(v)
            elif k == "height" and v.lstrip("-").isdigit():
                info["height"] = int(v)
            elif k == "duration":
                try:
                    info["duration"] = float(v)
                except ValueError:
                    pass
            elif k == "TAG:rotate" and v.lstrip("-").isdigit():
                info["rotate_tag"] = int(v) % 360
            elif k == "rotation" and v.lstrip("-").isdigit():
                info["display_matrix_rotation"] = (-int(v)) % 360
        try:
            os.unlink(stderr_tmp)
        except OSError:
            pass
    except Exception:
        pass
    return info


def prepare_browser_preview(input_path: str, out_path: str) -> bool:
    """
    Re-encode input_path to a browser-compatible H.264 MP4 saved at out_path.
    Caps preview at 480p to keep RAM/disk low on Streamlit Cloud.
    Returns True on success. Never loads video into Python memory.
    """
    stderr_tmp = tempfile.mktemp(suffix=".txt")
    try:
        vid_info = _get_video_info(input_path)
        rotation = vid_info["rotate_tag"] or vid_info["display_matrix_rotation"]

        if rotation == 90:
            rotate_filter = "transpose=1,"
        elif rotation == 270:
            rotate_filter = "transpose=2,"
        elif rotation == 180:
            rotate_filter = "transpose=1,transpose=1,"
        else:
            rotate_filter = ""

        # Cap preview at 480p — saves RAM and /tmp space
        # Simple: scale longest side to 854, force even dims
        scale_480 = "scale=640:360:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
        vf_plain        = scale_480
        vf_with_rotate  = f"{rotate_filter}{scale_480}"

        encode_flags = ["-c:v", "libx264", "-crf", "26",
                        "-preset", "fast", "-pix_fmt", "yuv420p"]

        def _run(cmd):
            with open(stderr_tmp, "wb") as ef:
                return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                      stderr=ef, timeout=300)

        def _ok(proc):
            return (proc.returncode == 0 and os.path.exists(out_path)
                    and os.path.getsize(out_path) > 0)

        r = _run(["ffmpeg", "-y", "-ignore_editlist", "1",
                  "-i", input_path, *encode_flags,
                  "-vf", vf_plain, "-movflags", "+faststart", "-an", out_path])

        if not _ok(r):
            r = _run(["ffmpeg", "-y", "-ignore_editlist", "1",
                      "-noautorotate", "-i", input_path, *encode_flags,
                      "-vf", vf_with_rotate, "-metadata:s:v:0", "rotate=0",
                      "-movflags", "+faststart", "-an", out_path])

        if not _ok(r):
            r = _run(["ffmpeg", "-y", "-ignore_editlist", "1",
                      "-i", input_path, "-c:v", "copy",
                      "-movflags", "+faststart", "-an", out_path])

        if not _ok(r):
            with open(stderr_tmp, "r", errors="replace") as ef:
                err_text = ef.read()[-2000:]
            st.error(
                f"⚠️ ffmpeg could not prepare preview (exit {r.returncode}). "
                f"**Run Motion Magnification** will still work.\n\n"
                f"```\n{err_text}\n```\n\n"
                f"**Codec:** `{vid_info['codec']}` · **Rotation:** `{rotation}°`"
            )
            return False
        return True

    except FileNotFoundError:
        st.error("ffmpeg not found — add `ffmpeg` to packages.txt.")
        return False
    except subprocess.TimeoutExpired:
        st.error("ffmpeg timed out preparing preview. Try a shorter clip.")
        return False
    finally:
        try:
            os.unlink(stderr_tmp)
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

    st.markdown("#### 📐 Resolution Cap")
    st.markdown(
        '<div class="param-label">Max dimension (px) — reduces RAM & /tmp usage</div>',
        unsafe_allow_html=True,
    )
    res_options = {"No cap (full resolution)": None, "1080p": 1080, "720p": 720, "480p": 480, "360p": 360}
    res_label = st.selectbox("Max Dimension", list(res_options.keys()), index=0,
                             label_visibility="collapsed")
    max_dimension = res_options[res_label]

    st.markdown("---")
    st.markdown("#### 🌊 Vibration Analysis")
    st.markdown('<div class="param-label">Optical Flow Method</div>', unsafe_allow_html=True)
    of_method = st.selectbox("Optical Flow Method", ["farneback", "lucas_kanade"],
                             label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">STEP 1 — Upload &amp; Magnify</div>', unsafe_allow_html=True)
st.markdown("### 📤 Upload Video")

uploaded_file = st.file_uploader(
    "Drop your video here (MP4, MOV, AVI, MKV — including iPhone HEVC)",
    type=["mp4", "avi", "mov", "mkv", "hevc", "m4v"],
)

tmp_input_path = None
vid_w = vid_h = total_frames = 0
vid_fps = 0.0

if uploaded_file:
    try:
        fname = uploaded_file.name
        file_size_mb = uploaded_file.size / (1024 * 1024)

        if st.session_state["last_upload_name"] != fname:
            raw_bytes = uploaded_file.read()
            suffix = os.path.splitext(fname)[1].lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                st.session_state["orig_tmp_path"] = tmp.name
            del raw_bytes
            st.session_state["last_upload_name"] = fname
            st.session_state["orig_preview_path"] = None
            st.session_state["magnified_path"]   = None
            st.session_state["mag_vid_bytes"]    = None
            st.session_state["confirmed_roi"]    = None
            st.session_state["pending_roi"]      = None

        tmp_input_path = st.session_state["orig_tmp_path"]

        vid_info = _get_video_info(tmp_input_path)
        is_hevc  = vid_info["codec"] in ("hevc", "h265", "h.265")
        rotation = vid_info["rotate_tag"] or vid_info["display_matrix_rotation"]

        if st.session_state["orig_preview_path"] is None:
            codec_label = vid_info["codec"].upper() if vid_info["codec"] != "unknown" else "video"
            with st.spinner(
                f"Preparing {codec_label} video for playback"
                + (" (iPhone HEVC — this may take a moment)…" if is_hevc else "…")
            ):
                preview_path = tmp_input_path + "_preview.mp4"
                ok = prepare_browser_preview(tmp_input_path, preview_path)
                st.session_state["orig_preview_path"] = preview_path if ok else None

        if vid_info["width"] > 0 and vid_info["height"] > 0:
            if rotation in (90, 270):
                vid_w, vid_h = vid_info["height"], vid_info["width"]
            else:
                vid_w, vid_h = vid_info["width"], vid_info["height"]
            total_frames = 0
            vid_fps = 0.0

        cap = cv2.VideoCapture(tmp_input_path)
        if cap.isOpened():
            if total_frames == 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if vid_fps == 0.0:
                vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_w == 0:
                vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if vid_h == 0:
                vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

        duration = total_frames / vid_fps if vid_fps > 0 else vid_info["duration"]

        left_up, right_up = st.columns([1, 1], gap="large")

        with left_up:
            st.markdown("**Original**")

            if is_hevc:
                st.info("📱 iPhone HEVC (H.265) detected — transcoding to H.264 for preview")
            elif rotation:
                st.info(f"🔄 Rotation detected: {rotation}° — baked into preview")

            orig_preview = st.session_state["orig_preview_path"]
            if orig_preview and os.path.exists(orig_preview):
                with open(orig_preview, "rb") as _pf:
                    st.video(_pf.read(), format="video/mp4")
            else:
                st.warning(
                    "⚠️ Preview unavailable — ffmpeg could not re-encode this video. "
                    "The file is still uploaded and can be processed below."
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Resolution", f"{vid_w}×{vid_h}")
            m2.metric("FPS", f"{vid_fps:.0f}" if vid_fps else "—")
            m3.metric("Duration", f"{duration:.1f}s" if duration else "—")
            m4.metric("Size", f"{file_size_mb:.1f} MB")

        with right_up:
            st.markdown("**Magnified output**")

            if max_dimension and max(vid_w, vid_h) > max_dimension:
                scale  = max_dimension / max(vid_w, vid_h)
                proc_w = int(vid_w * scale) & ~1
                proc_h = int(vid_h * scale) & ~1
                st.caption(f"ℹ️ Will process at {proc_w}×{proc_h} (capped to {max_dimension}p)")
            elif vid_w and vid_h:
                st.caption(f"ℹ️ Will process at full {vid_w}×{vid_h}")

            run_btn = st.button("🚀 Run Motion Magnification", use_container_width=True)




            if run_btn:
                st.session_state["magnified_path"] = None
                st.session_state["mag_vid_bytes"]  = None
                st.session_state["confirmed_roi"]  = None
                st.session_state["pending_roi"]    = None

                out_path = tmp_input_path.replace(
                    os.path.splitext(tmp_input_path)[1], "_magnified.mp4"
                )

                # ── Rotation fix: bake rotation into temp file so OpenCV sees correct orientation ──
                rotation = vid_info["rotate_tag"] or vid_info["display_matrix_rotation"]
                if rotation in (90, 180, 270):
                    rotated_input = tmp_input_path.replace(
                        os.path.splitext(tmp_input_path)[1], "_rotfix.mp4"
                    )
                    if rotation == 90:
                        vf = "transpose=1"
                    elif rotation == 270:
                        vf = "transpose=2"
                    elif rotation == 180:
                        vf = "transpose=1,transpose=1"
                    subprocess.run([
                        "ffmpeg", "-y", "-i", tmp_input_path,
                        "-vf", vf,
                        "-metadata:s:v:0", "rotate=0",
                        "-c:v", "libx264", "-crf", "18",
                        "-preset", "fast", "-pix_fmt", "yuv420p",
                        rotated_input
                    ], check=True)
                    processing_input = rotated_input
                else:
                    processing_input = tmp_input_path
                # ────────────────────────────────────────────────────────────────────────────

                status   = st.empty()
                prog     = st.progress(0)
                eta_slot = st.empty()
                t0 = time.time()

                def _cb(cur, tot):
                    prog.progress(min(cur / tot, 1.0))
                    el = time.time() - t0
                    if cur > 1:
                        eta_slot.caption(
                            f"Frame {cur}/{tot} · ETA {(el/(cur/tot))*(1-cur/tot):.0f}s"
                        )

                try:
                    status.info("🔄 Processing frames… (piping directly to H.264)")
                    final_path = process_video(
                        input_path=processing_input,
                        output_path=out_path,
                        alpha=alpha,
                        lambda_c=lambda_c,
                        fl=fl,
                        fh=fh,
                        fps=fps_sidebar,
                        progress_callback=_cb,
                        n_levels=n_levels,
                        alpha_curve=alpha_curve,
                        max_dimension=max_dimension,
                    )
                    prog.progress(1.0)
                    eta_slot.empty()

                    status.info("📦 Reading output…")
                    with open(final_path, "rb") as f:
                        mag_bytes = f.read()

                    cap_out = cv2.VideoCapture(final_path)
                    out_w   = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
                    out_h   = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out_fps = cap_out.get(cv2.CAP_PROP_FPS)
                    cap_out.release()

                    st.session_state["mag_vid_bytes"]  = mag_bytes
                    st.session_state["magnified_path"] = final_path
                    st.session_state["mag_vid_w"]      = out_w
                    st.session_state["mag_vid_h"]      = out_h
                    st.session_state["mag_vid_fps"]    = out_fps if out_fps > 0 else float(fps_sidebar)
                    status.success(f"✅ Done in {time.time()-t0:.1f}s — output: {out_w}×{out_h}")

                except Exception as e:
                    status.error(f"❌ {e}")
                    prog.empty()
                    eta_slot.empty()

            mag_bytes = st.session_state.get("mag_vid_bytes")
            if mag_bytes:
                st.video(mag_bytes, format="video/mp4")
                st.download_button(
                    "⬇️ Download Magnified Video", data=mag_bytes,
                    file_name="magnified_output.mp4", mime="video/mp4",
                    use_container_width=True,
                )

    except Exception as _top_err:
        st.error(f"**Unhandled exception — please share this with the developer:**\n```\n{traceback.format_exc()}\n```")

else:
    for k in ["orig_tmp_path", "orig_preview_path", "magnified_path",
              "mag_vid_bytes", "confirmed_roi", "pending_roi", "last_upload_name"]:
        st.session_state[k] = None
    st.info("Upload a video above, tune parameters in the sidebar, then click **Run Motion Magnification**.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ROI canvas
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

    cap_m = cv2.VideoCapture(mag_path)
    ret_m, first_frame = cap_m.read()
    cap_m.release()

    if not ret_m:
        st.warning("Could not read first frame of magnified video.")
    else:
        CANVAS_W = min(700, m_w) if m_w > 0 else 700
        CANVAS_H = int(m_h * CANVAS_W / m_w) if m_w > 0 else 400

        _, buf = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_frame = base64.b64encode(buf.tobytes()).decode()

        qp = st.query_params
        if "roi" in qp:
            try:
                parts = [int(v) for v in qp["roi"].split(",")]
                if len(parts) == 4 and parts[2] >= 4 and parts[3] >= 4:
                    st.session_state["pending_roi"] = tuple(parts)
            except Exception:
                pass

        pending = st.session_state.get("pending_roi")
        if pending:
            st.info(f"📐 Drawn — x:{pending[0]}  y:{pending[1]}  w:{pending[2]}  h:{pending[3]}  ← click Confirm ROI to lock")

        canvas_html = f"""
<style>
  body {{ margin:0; background:transparent; font-family:'Space Mono',monospace; }}
  canvas {{ display:block; cursor:crosshair; border-radius:8px;
            border:2px solid #2d2d4e; max-width:100%; }}
  #hint {{ font-size:0.7rem; color:#6b7280; margin:5px 0 6px; }}
  #lbl  {{ font-size:0.75rem; color:#38bdf8; background:#13131f;
           border:1px solid #1e1e30; border-radius:5px;
           padding:4px 10px; display:inline-block; margin-bottom:6px; }}
  #btn  {{ background:linear-gradient(135deg,#7c3aed,#2563eb); color:#fff;
           border:none; border-radius:7px; font-family:'Syne',sans-serif;
           font-weight:700; font-size:0.85rem; padding:8px 26px;
           cursor:pointer; margin-top:6px; }}
  #btn.ok {{ background:linear-gradient(135deg,#059669,#0284c7); }}
</style>
<canvas id="c" width="{CANVAS_W}" height="{CANVAS_H}"></canvas>
<div id="hint">Drag to draw ROI, then click Confirm ROI.</div>
<div id="lbl">No ROI drawn yet</div><br>
<button id="btn" onclick="confirm_roi()">✅ Confirm ROI</button>
<script>
const C   = document.getElementById('c');
const ctx = C.getContext('2d');
const NW={CANVAS_W}, NH={CANVAS_H}, VW={m_w}, VH={m_h};
const img = new Image();
img.onload = () => ctx.drawImage(img, 0, 0, NW, NH);
img.src = 'data:image/jpeg;base64,{b64_frame}';

function pt(e) {{
  const r = C.getBoundingClientRect();
  const s = e.touches ? e.touches[0] : e;
  return [(s.clientX - r.left) * (NW / r.width),
          (s.clientY - r.top)  * (NH / r.height)];
}}
let drawing=false, sx=0, sy=0, box={{}}, done=false;
function draw_box(x,y,w,h) {{
  ctx.save();
  ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2; ctx.setLineDash([5,3]);
  ctx.strokeRect(x,y,w,h);
  ctx.fillStyle='rgba(245,158,11,0.12)'; ctx.fillRect(x,y,w,h);
  ctx.restore();
}}
function redraw(ex,ey) {{
  ctx.clearRect(0,0,NW,NH); ctx.drawImage(img,0,0,NW,NH);
  draw_box(Math.min(sx,ex),Math.min(sy,ey),
           Math.abs(ex-sx),Math.abs(ey-sy));
}}
function to_vid(cx,cy,ex,ey) {{
  return {{
    x: Math.round(Math.min(cx,ex)*VW/NW),
    y: Math.round(Math.min(cy,ey)*VH/NH),
    w: Math.round(Math.abs(ex-cx)*VW/NW),
    h: Math.round(Math.abs(ey-cy)*VH/NH),
  }};
}}
function set_label(b) {{
  document.getElementById('lbl').innerText =
    'ROI → x:'+b.x+' y:'+b.y+' w:'+b.w+' h:'+b.h;
}}
C.addEventListener('mousedown', e=>{{
  [sx,sy]=pt(e); drawing=true; done=false; e.preventDefault(); }});
C.addEventListener('mousemove', e=>{{
  if(!drawing) return;
  const [ex,ey]=pt(e); redraw(ex,ey);
  set_label(to_vid(sx,sy,ex,ey)); e.preventDefault(); }});
C.addEventListener('mouseup', e=>{{
  if(!drawing) return; drawing=false;
  const [ex,ey]=pt(e); redraw(ex,ey);
  box=to_vid(sx,sy,ex,ey); set_label(box); done=true; e.preventDefault(); }});
C.addEventListener('touchstart', e=>{{
  [sx,sy]=pt(e); drawing=true; done=false; e.preventDefault(); }},{{passive:false}});
C.addEventListener('touchmove', e=>{{
  if(!drawing) return;
  const [ex,ey]=pt(e); redraw(ex,ey); e.preventDefault(); }},{{passive:false}});
C.addEventListener('touchend', e=>{{
  if(!drawing) return; drawing=false;
  const [ex,ey]=pt(e);
  box=to_vid(sx,sy,ex,ey); set_label(box); done=true; e.preventDefault(); }},{{passive:false}});

function confirm_roi() {{
  if(!done || box.w<4 || box.h<4) {{ alert('Draw a larger box first.'); return; }}
  const val = box.x+','+box.y+','+box.w+','+box.h;
  const url = new URL(window.parent.location.href);
  url.searchParams.set('roi', val);
  window.parent.history.pushState({{}}, '', url);
  window.parent.dispatchEvent(new Event('popstate'));
  document.getElementById('btn').className = 'ok';
  document.getElementById('btn').innerText = '✅ ROI sent — click Confirm ROI below';
}}
</script>
"""
        components.html(canvas_html, height=CANVAS_H + 140, scrolling=False)

        col_confirm, col_clear = st.columns([3, 1])
        with col_confirm:
            if st.button("✅ Confirm ROI", use_container_width=True):
                qp2 = st.query_params
                if "roi" in qp2:
                    try:
                        parts = [int(v) for v in qp2["roi"].split(",")]
                        if len(parts) == 4 and parts[2] >= 4 and parts[3] >= 4:
                            st.session_state["confirmed_roi"] = tuple(parts)
                            st.session_state["pending_roi"]   = tuple(parts)
                            st.query_params.clear()
                            st.success(
                                f"✅ ROI confirmed — x:{parts[0]}  y:{parts[1]}"
                                f"  w:{parts[2]}  h:{parts[3]}"
                            )
                        else:
                            st.warning("ROI too small — draw a larger box.")
                    except Exception:
                        st.warning("Could not parse ROI. Please draw again.")
                elif st.session_state.get("pending_roi"):
                    st.session_state["confirmed_roi"] = st.session_state["pending_roi"]
                    p = st.session_state["confirmed_roi"]
                    st.success(f"✅ ROI confirmed — x:{p[0]}  y:{p[1]}  w:{p[2]}  h:{p[3]}")
                else:
                    st.warning("Draw a rectangle on the canvas first.")
        with col_clear:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state["confirmed_roi"] = None
                st.session_state["pending_roi"]   = None
                st.query_params.clear()

        confirmed = st.session_state.get("confirmed_roi")
        if confirmed:
            rx, ry, rw, rh = confirmed
            preview = first_frame.copy()
            overlay = preview.copy()
            cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (245, 158, 11), -1)
            preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)
            cv2.rectangle(preview, (rx, ry), (rx+rw, ry+rh), (245, 158, 11), 2)
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
