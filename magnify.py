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
    "auto":      None,          # original spatial-frequency-based attenuation
    "flat":      "flat",        # same alpha at every level
    "linear":    "linear",      # ramps from 0 at coarse → alpha at fine
    "quadratic": "quadratic",   # emphasises fine detail more aggressively
    "inverse":   "inverse",     # opposite: coarse levels get more weight
}


class Magnify(object):
    def __init__(self, img1, alpha, lambda_c, fl, fh, samplingRate,
                 n_levels=None, alpha_curve="auto"):
        """
        Parameters
        ----------
        img1        : first BGR frame (uint8)
        alpha       : magnification strength
        lambda_c    : spatial wavelength cutoff (pixels)
        fl / fh     : temporal bandpass bounds (Hz)
        samplingRate: video fps
        n_levels    : number of Laplacian pyramid levels (None = pyrtools default)
        alpha_curve : how alpha is distributed across pyramid levels
                      one of "auto", "flat", "linear", "quadratic", "inverse"
        """
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

            # optionally truncate / pad levels
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
        """Return effective alpha for pyramid level l (0 = finest)."""
        t = l / max(n_levels - 1, 1)          # 0 (finest) … 1 (coarsest)
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
            return None                        # "auto" — use original spatial logic

    def process_frame(self, img2):
        img2 = img_as_float(img2)
        output_channels = []

        for c in range(3):
            py2 = pt.pyramids.LaplacianPyramid(img2[:, :, c])
            py2._build_pyr()
            full_pyr = list(py2.pyr_coeffs.values())

            n_levels = len(self.pyramids[c])
            pyr = full_pyr[:n_levels]           # match stored depth

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
                    # original auto spatial-frequency attenuation
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
#  Video processing
# ─────────────────────────────────────────────────────────────────────────────

def _remux_h264(src_path: str, dst_path: str) -> bool:
    """Re-encode src (mp4v) to H.264 using ffmpeg so browsers can play it."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                dst_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return result.returncode == 0
    except Exception:
        return False


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
):
    """
    Process a video file with motion magnification.

    Returns
    -------
    str : path of the final (browser-playable) output file
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write raw frames with mp4v first (universally writable via OpenCV)
    raw_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    ret, img1 = cap.read()
    if not ret:
        cap.release(); out.release()
        raise ValueError("Failed to read first frame from video.")

    magnifier = Magnify(
        img1, alpha, lambda_c, fl, fh, fps,
        n_levels=n_levels,
        alpha_curve=alpha_curve,
    )
    out.write(img1)

    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(magnifier.process_frame(frame))
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, total_frames)

    cap.release()
    out.release()

    # Re-encode to H.264 so st.video() can play it inline
    if _remux_h264(raw_path, output_path):
        try:
            os.remove(raw_path)
        except OSError:
            pass
        return output_path
    else:
        # ffmpeg not available – fall back to raw file
        try:
            os.rename(raw_path, output_path)
        except OSError:
            pass
        return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  ROI optical-flow + FFT vibration analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_vibration(
    input_path: str,
    roi: tuple,           # (x, y, w, h) in pixels
    fps: float,
    method: str = "farneback",   # "farneback" | "lucas_kanade"
    progress_callback=None,
) -> dict:
    """
    Extract per-frame motion magnitude inside *roi*, then compute FFT to find
    dominant vibration frequencies.

    Parameters
    ----------
    input_path : path to source video
    roi        : (x, y, w, h) bounding box
    fps        : frames per second
    method     : optical-flow method
    progress_callback : callable(current, total)

    Returns
    -------
    dict with keys:
        times       – 1-D array, seconds
        motion      – 1-D array, mean optical-flow magnitude per frame (px/frame)
        freqs       – FFT frequency axis (Hz), positive half
        power       – FFT power spectrum (magnitude²)
        dominant_hz – frequency with highest power (Hz)
        dominant_amp– amplitude at dominant frequency
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
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5, levels=3, winsize=13,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_signal.append(float(mag.mean()))

        else:  # lucas_kanade
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

    # Remove DC offset before FFT
    motion_ac = motion - motion.mean()

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(N)
    fft_vals = np.fft.rfft(motion_ac * window)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    power = (np.abs(fft_vals) ** 2) / N

    # Ignore DC bin
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
