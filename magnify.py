import numpy as np
import scipy.signal as signal
import pyrtools as pt
import copy
import cv2
from skimage import img_as_float, img_as_ubyte
import subprocess
import os


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
#  FIX: Direct ffmpeg pipe — no intermediate raw file written to disk
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
    max_dimension=None,   # e.g. 720 — downscale if larger
):
    """
    Process a video file with motion magnification.

    Key fix: frames are piped DIRECTLY into ffmpeg (libx264) — no huge
    intermediate raw .mp4 file is written to disk, saving hundreds of MB
    of /tmp space on memory-constrained hosts (Streamlit Cloud etc.).

    Parameters
    ----------
    max_dimension : int or None
        If set, the larger of (width, height) is capped at this value before
        processing.  Useful for high-res uploads that would otherwise exhaust
        RAM / /tmp.  e.g. 720 keeps most detail while cutting memory ~4×.

    Returns
    -------
    str : path of the final (browser-playable H.264) output file
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Optional downscale ────────────────────────────────────────────────────
    if max_dimension and max(orig_w, orig_h) > max_dimension:
        scale = max_dimension / max(orig_w, orig_h)
        # ffmpeg needs even dimensions for yuv420p
        w = int(orig_w * scale) & ~1
        h = int(orig_h * scale) & ~1
    else:
        w = orig_w & ~1   # ensure even for yuv420p
        h = orig_h & ~1

    # ── Spin up ffmpeg, reading raw BGR frames from stdin ────────────────────
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
        output_path,
    ]

    try:
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,   # capture errors only (not stdout)
        )
    except FileNotFoundError:
        cap.release()
        raise RuntimeError(
            "ffmpeg not found. Add `ffmpeg` to packages.txt in your repo root."
        )

    def _write_frame(frame):
        """Resize if needed, then pipe raw bytes to ffmpeg."""
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ffmpeg_proc.stdin.write(frame.tobytes())

    try:
        ret, img1 = cap.read()
        if not ret:
            raise ValueError("Failed to read first frame from video.")

        magnifier = Magnify(
            img1 if (img1.shape[1] == w and img1.shape[0] == h)
                else cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA),
            alpha, lambda_c, fl, fh, fps,
            n_levels=n_levels,
            alpha_curve=alpha_curve,
        )
        _write_frame(img1)

        frame_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = magnifier.process_frame(
                frame if (frame.shape[1] == w and frame.shape[0] == h)
                      else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            )
            _write_frame(processed)
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)

    finally:
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
        except BrokenPipeError:
            pass
        _, stderr_bytes = ffmpeg_proc.communicate()
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed:\n{stderr_bytes.decode(errors='replace')}"
            )

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  ROI optical-flow + FFT vibration analysis  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_vibration(
    input_path: str,
    roi: tuple,
    fps: float,
    method: str = "farneback",
    progress_callback=None,
) -> dict:
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
                    mag = np.sqrt((diff ** 2).sum(axis=1))
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
    N = len(motion)
    times = np.arange(N) / fps
    motion_ac = motion - motion.mean()
    window = np.hanning(N)
    fft_vals = np.fft.rfft(motion_ac * window)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    power = (np.abs(fft_vals) ** 2) / N

    if len(freqs) > 1:
        freqs = freqs[1:]
        power = power[1:]

    dom_idx = int(np.argmax(power))
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
