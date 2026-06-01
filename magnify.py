import numpy as np
import scipy.signal as signal
import pyrtools as pt
import copy
import cv2
from skimage import img_as_float, img_as_ubyte
import subprocess
import os
import tempfile


def reconPyr(pyr):
    filt2 = pt.binomial_filter(5)
    maxLev = len(pyr)
    res = []

    for lev in range(maxLev - 1, -1, -1):
        if len(res) == 0:
            res = pyr[lev]
        else:
            res_sz = res.shape
            new_sz = pyr[lev].shape
            if res_sz[0] == 1:
                hi2 = pt.upConv(image=res, filt=filt2, step=(2, 1), stop=(new_sz[1], new_sz[0])).T
            elif res_sz[1] == 1:
                hi2 = pt.upConv(image=res, filt=filt2.T, step=(1, 2), stop=(new_sz[1], new_sz[0])).T
            else:
                hi = pt.upConv(image=res, filt=filt2, step=(2, 1), stop=(new_sz[0], res_sz[1]))
                hi2 = pt.upConv(image=hi, filt=filt2.T, step=(1, 2), stop=(new_sz[0], new_sz[1]))
            bandIm = pyr[lev]
            res = hi2 + bandIm
    return res


ALPHA_CURVES = {
    "auto":      None,
    "flat":      "flat",
    "linear":    "linear",
    "quadratic": "quadratic",
    "inverse":   "inverse",
}


class Magnify(object):
    def __init__(self, img1, alpha, lambda_c, fl, fh, samplingRate,
                 n_levels=None, alpha_curve="auto"):
        [low_a, low_b] = signal.butter(1, fl / samplingRate, 'low')
        [high_a, high_b] = signal.butter(1, fh / samplingRate, 'low')

        self.pyramids  = []
        self.lowpass1  = []
        self.lowpass2  = []
        self.filtered  = []

        img1 = img_as_float(img1)

        for i in range(3):
            py1 = pt.pyramids.LaplacianPyramid(img1[:, :, i])
            py1._build_pyr()
            pyramid_1 = list(py1.pyr_coeffs.values())

            if n_levels is not None:
                n_levels_clamped = max(1, min(n_levels, len(pyramid_1)))
                pyramid_1 = pyramid_1[:n_levels_clamped]

            self.pyramids.append(pyramid_1)
            nLevels = len(pyramid_1)
            self.lowpass1.append([np.zeros_like(pyramid_1[j]) for j in range(nLevels)])
            self.lowpass2.append([np.zeros_like(pyramid_1[j]) for j in range(nLevels)])
            self.filtered.append([None for _ in range(nLevels)])

        self.alpha        = alpha
        self.alpha_curve  = alpha_curve
        self.fl           = fl
        self.fh           = fh
        self.samplingRate = samplingRate
        self.low_a  = low_a;  self.low_b  = low_b
        self.high_a = high_a; self.high_b = high_b
        self.width  = img1.shape[0]
        self.height = img1.shape[1]
        self.lambd  = (self.width ** 2 + self.height ** 2) / 3.0
        self.lambda_c = lambda_c
        self.delta  = self.lambda_c / 8.0 / (1 + self.alpha)

    def _level_alpha(self, l, n_levels):
        t = l / max(n_levels - 1, 1)
        curve = self.alpha_curve
        if curve == "flat":
            return self.alpha
        elif curve == "linear":
            return self.alpha * (1.0 - t)
        elif curve == "quadratic":
            return self.alpha * (1.0 - t) ** 2
        elif curve == "inverse":
            return self.alpha * t
        else:
            return None

    def process_frame(self, img2):
        img2 = img_as_float(img2)
        output_channels = []

        for c in range(3):
            py2 = pt.pyramids.LaplacianPyramid(img2[:, :, c])
            py2._build_pyr()
            full_pyr = list(py2.pyr_coeffs.values())

            n_levels = len(self.pyramids[c])
            pyr = full_pyr[:n_levels]

            for u in range(n_levels):
                self.lowpass1[c][u] = (
                    -self.high_b[1] * self.lowpass1[c][u]
                    + self.high_a[0] * pyr[u]
                    + self.high_a[1] * self.pyramids[c][u]
                ) / self.high_b[0]
                self.lowpass2[c][u] = (
                    -self.low_b[1] * self.lowpass2[c][u]
                    + self.low_a[0] * pyr[u]
                    + self.low_a[1] * self.pyramids[c][u]
                ) / self.low_b[0]
                self.filtered[c][u] = self.lowpass1[c][u] - self.lowpass2[c][u]

            self.pyramids[c] = copy.deepcopy(pyr)

            exaggeration_factor = 2
            lambd = self.lambd
            delta = self.delta
            filtered = self.filtered[c]

            for l in range(len(filtered) - 1, -1, -1):
                custom_alpha = self._level_alpha(l, n_levels)

                if l == len(filtered) - 1 or l == 0:
                    filtered[l] = np.zeros_like(filtered[l])
                elif custom_alpha is None:
                    currAlpha = lambd / delta / 8.0 - 1
                    currAlpha *= exaggeration_factor
                    if currAlpha > self.alpha:
                        filtered[l] = self.alpha * filtered[l]
                    else:
                        filtered[l] = currAlpha * filtered[l]
                else:
                    filtered[l] = custom_alpha * filtered[l]

                lambd /= 2.0

            output_channels.append(reconPyr(filtered))

        output = np.stack(output_channels, axis=2)
        output = img2 + output
        output = np.clip(output, 0, 1)
        return img_as_ubyte(output)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _even(n):
    """Round down to nearest even number, minimum 2 (never produces 0)."""
    return max(2, int(n) & ~1)


def _detect_rotation(input_path: str) -> int:
    """
    Read the video stream's rotation via ffprobe.
    Returns 0, 90, 180, or 270.

    Checks both:
    1. Simple 'rotate' stream tag  (Android, older cameras, many MP4s).
    2. 'side_data' display matrix  (iPhone iOS 14+, newer .mov files).

    Always reads from the ORIGINAL file path — never from an interim
    transcode — so the rotation tag is never missing.
    """
    stderr_tmp = None

    # ── Method 1: rotate tag ──────────────────────────────────────────────────
    try:
        stderr_tmp = tempfile.mktemp(suffix=".txt")
        with open(stderr_tmp, "wb") as ef:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream_tags=rotate",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    input_path,
                ],
                stdout=subprocess.PIPE,
                stderr=ef,
                timeout=15,
            )
        val = probe.stdout.decode(errors="replace").strip()
        if val and val.lstrip("-").isdigit():
            return int(val) % 360
    except Exception:
        pass
    finally:
        if stderr_tmp:
            try:
                os.unlink(stderr_tmp)
            except OSError:
                pass
        stderr_tmp = None

    # ── Method 2: display matrix (iPhone iOS 14+) ─────────────────────────────
    try:
        stderr_tmp = tempfile.mktemp(suffix=".txt")
        with open(stderr_tmp, "wb") as ef:
            probe2 = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream_side_data=rotation",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    input_path,
                ],
                stdout=subprocess.PIPE,
                stderr=ef,
                timeout=15,
            )
        val2 = probe2.stdout.decode(errors="replace").strip()
        if val2 and val2.lstrip("-").isdigit():
            return (-int(val2)) % 360
    except Exception:
        pass
    finally:
        if stderr_tmp:
            try:
                os.unlink(stderr_tmp)
            except OSError:
                pass

    return 0


def _detect_codec(input_path: str) -> str:
    """
    Return the video stream codec name (e.g. 'hevc', 'h264', 'mpeg4').
    Returns 'unknown' on failure.
    """
    stderr_tmp = None
    try:
        stderr_tmp = tempfile.mktemp(suffix=".txt")
        with open(stderr_tmp, "wb") as ef:
            r = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    input_path,
                ],
                stdout=subprocess.PIPE,
                stderr=ef,
                timeout=15,
            )
        codec = r.stdout.decode(errors="replace").strip().lower()
        return codec if codec else "unknown"
    except Exception:
        return "unknown"
    finally:
        if stderr_tmp:
            try:
                os.unlink(stderr_tmp)
            except OSError:
                pass


def _apply_rotation(frame: np.ndarray, rot: int) -> np.ndarray:
    """Rotate a BGR frame to match EXIF/container orientation."""
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _open_video_capture(input_path: str, codec: str) -> tuple:
    """
    Open a VideoCapture, with a special pre-transcode path for HEVC/H.265
    (iPhone) videos that OpenCV cannot decode natively.

    IMPORTANT: the MJPEG transcode uses -noautorotate so ffmpeg does NOT
    bake the rotation into the pixel data.  Rotation is handled separately
    by _apply_rotation(), which reads the tag from the ORIGINAL file.
    Without -noautorotate, ffmpeg auto-rotates during transcode AND
    _apply_rotation rotates again → double rotation → wrong orientation.

    Returns (cap, temp_path_or_None).  Caller must release cap and, if
    temp_path is not None, delete it when done.
    """
    if codec in ("hevc", "h265", "h.265"):
        tmp_path = tempfile.mktemp(suffix="_interim.avi")
        stderr_tmp = tempfile.mktemp(suffix=".txt")
        try:
            with open(stderr_tmp, "wb") as ef:
                r = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-noautorotate",       # ← KEY FIX: do NOT bake rotation
                                               #   into pixel data; _apply_rotation
                                               #   handles it from the original tag
                        "-ignore_editlist", "1",
                        "-i", input_path,
                        "-c:v", "mjpeg",
                        "-q:v", "3",
                        "-an",
                        tmp_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=ef,
                    timeout=300,
                )
            if (
                r.returncode == 0
                and os.path.exists(tmp_path)
                and os.path.getsize(tmp_path) > 0
            ):
                cap = cv2.VideoCapture(tmp_path)
                if cap.isOpened():
                    return cap, tmp_path
                cap.release()
        except Exception:
            pass
        finally:
            try:
                os.unlink(stderr_tmp)
            except OSError:
                pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    cap = cv2.VideoCapture(input_path)
    return cap, None


# ─────────────────────────────────────────────────────────────────────────────
#  Video processing — direct ffmpeg pipe, portrait-safe, HEVC-safe
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path,
    output_path,
    alpha,
    lambda_c,
    fl,
    fh,
    fps,
    progress_callback=None,
    n_levels=None,
    alpha_curve="auto",
    max_dimension=None,
):
    """
    Process a video file with Eulerian motion magnification.

    Rotation handling
    -----------------
    Rotation is ALWAYS detected from the original input_path (before any
    HEVC transcode).  The HEVC→MJPEG transcode is done with -noautorotate
    so ffmpeg never touches the pixel orientation — _apply_rotation() is
    the single, authoritative place that corrects the frame.

    This prevents the double-rotation bug where:
      1. ffmpeg auto-rotates during HEVC transcode
      2. _apply_rotation rotates again
      → output is rotated 90° the wrong way
    """
    # ── Detect codec and rotation from the ORIGINAL file ─────────────────────
    # Must happen before _open_video_capture so we always read the real tag,
    # not a stripped interim file.
    codec    = _detect_codec(input_path)
    rotation = _detect_rotation(input_path)   # reads from original, always correct

    # ── Open VideoCapture (HEVC transcode uses -noautorotate) ─────────────────
    cap, interim_path = _open_video_capture(input_path, codec)

    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video: {input_path} "
            f"(codec={codec}, rotation={rotation}°)"
        )

    try:
        raw_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # After applying rotation, logical frame dimensions may be swapped.
        # For HEVC with -noautorotate the interim file still has raw (unrotated)
        # dimensions, so raw_w/raw_h are the pre-rotation values — correct.
        if rotation in (90, 270):
            orig_w, orig_h = raw_h, raw_w
        else:
            orig_w, orig_h = raw_w, raw_h

        # ── Optional downscale (applied to post-rotation dimensions) ──────────
        if max_dimension and max(orig_w, orig_h) > max_dimension:
            scale = max_dimension / max(orig_w, orig_h)
            w = _even(orig_w * scale)
            h = _even(orig_h * scale)
        else:
            w = _even(orig_w)
            h = _even(orig_h)

        # ── Prepare every frame: rotate then resize to (w, h) ─────────────────
        def _prepare(frame: np.ndarray) -> np.ndarray:
            if rotation:
                frame = _apply_rotation(frame, rotation)
            fh_fr, fw_fr = frame.shape[:2]
            if fw_fr != w or fh_fr != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return frame

        stderr_tmp  = tempfile.mktemp(suffix=".txt")
        stderr_file = None

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            # Rotation is baked into pixels — clear any residual tag so
            # players don't attempt a second rotation on playback.
            "-metadata:s:v:0", "rotate=0",
            output_path,
        ]

        try:
            stderr_file = open(stderr_tmp, "wb")
            try:
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "ffmpeg not found. Add `ffmpeg` to packages.txt in your repo root."
                )

            try:
                ret, raw1 = cap.read()
                if not ret:
                    raise ValueError("Failed to read first frame from video.")

                img1 = _prepare(raw1)

                magnifier = Magnify(
                    img1,
                    alpha, lambda_c, fl, fh, fps,
                    n_levels=n_levels,
                    alpha_curve=alpha_curve,
                )
                ffmpeg_proc.stdin.write(img1.tobytes())

                frame_count = 1
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    prepared  = _prepare(frame)
                    processed = magnifier.process_frame(prepared)
                    ffmpeg_proc.stdin.write(processed.tobytes())
                    frame_count += 1
                    if progress_callback:
                        progress_callback(frame_count, total_frames)

            finally:
                try:
                    ffmpeg_proc.stdin.close()
                except BrokenPipeError:
                    pass

            ffmpeg_proc.wait()
            stderr_file.close()
            stderr_file = None

            if ffmpeg_proc.returncode != 0:
                with open(stderr_tmp, "r", errors="replace") as ef:
                    err_text = ef.read()[-3000:]
                raise RuntimeError(f"ffmpeg encoding failed:\n{err_text}")

        finally:
            if stderr_file and not stderr_file.closed:
                stderr_file.close()
            try:
                os.unlink(stderr_tmp)
            except OSError:
                pass

    finally:
        cap.release()
        if interim_path:
            try:
                os.unlink(interim_path)
            except OSError:
                pass

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  ROI optical-flow + FFT vibration analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_vibration(
    input_path: str,
    roi: tuple,
    fps: float,
    method: str = "farneback",
    progress_callback=None,
) -> dict:
    """
    Compute an optical-flow-based motion signal inside an ROI on the
    magnified video, then FFT it to find dominant vibration frequency.

    The magnified output is always H.264 MP4 (written by process_video),
    so no special HEVC handling is needed here — VideoCapture opens it fine.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    x, y, rw, rh = roi
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Cannot read first frame.")

    def crop_gray(frame):
        region = frame[y: y + rh, x: x + rw]
        return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    prev_gray = crop_gray(prev_frame)
    motion_signal = []
    frame_idx = 1

    if method == "lucas_kanade":
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
        )
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = crop_gray(frame)

        if method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=13,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_signal.append(float(mag.mean()))
        else:
            if p0 is None or len(p0) == 0:
                motion_signal.append(0.0)
            else:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, p0, None, **lk_params
                )
                good_new = p1[st == 1] if p1 is not None else np.array([])
                good_old = p0[st == 1] if p1 is not None else np.array([])
                if len(good_new) > 0:
                    diff = good_new - good_old
                    mag  = np.sqrt((diff ** 2).sum(axis=1))
                    motion_signal.append(float(mag.mean()))
                else:
                    motion_signal.append(0.0)
                p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else p0

        prev_gray = curr_gray
        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx, total_frames)

    cap.release()

    motion = np.array(motion_signal, dtype=float)
    N      = len(motion)
    times  = np.arange(N) / fps
    motion_ac = motion - motion.mean()
    window    = np.hanning(N)
    fft_vals  = np.fft.rfft(motion_ac * window)
    freqs     = np.fft.rfftfreq(N, d=1.0 / fps)
    power     = (np.abs(fft_vals) ** 2) / N

    # Remove DC component (0 Hz bin) — it's the static mean offset,
    # not a vibration frequency, and would otherwise dominate argmax.
    if len(freqs) > 1:
        freqs = freqs[1:]
        power = power[1:]

    dom_idx      = int(np.argmax(power))
    dominant_hz  = float(freqs[dom_idx])
    dominant_amp = float(np.sqrt(power[dom_idx]))

    return {
        "times":        times,
        "motion":       motion,
        "freqs":        freqs,
        "power":        power,
        "dominant_hz":  dominant_hz,
        "dominant_amp": dominant_amp,
    }
