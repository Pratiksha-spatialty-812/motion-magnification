import numpy as np
import scipy.signal as signal
import pyrtools as pt
import copy
import cv2
from skimage import img_as_float, img_as_ubyte


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


class Magnify(object):
    def __init__(self, img1, alpha, lambda_c, fl, fh, samplingRate):
        [low_a, low_b] = signal.butter(1, fl / samplingRate, 'low')
        [high_a, high_b] = signal.butter(1, fh / samplingRate, 'low')

        self.pyramids = []
        self.lowpass1 = []
        self.lowpass2 = []
        self.filtered = []

        img1 = img_as_float(img1)

        for i in range(3):
            py1 = pt.pyramids.LaplacianPyramid(img1[:, :, i])
            py1._build_pyr()
            pyramid_1 = list(py1.pyr_coeffs.values())
            self.pyramids.append(pyramid_1)
            nLevels = len(pyramid_1)
            self.lowpass1.append([np.zeros_like(pyramid_1[j]) for j in range(nLevels)])
            self.lowpass2.append([np.zeros_like(pyramid_1[j]) for j in range(nLevels)])
            self.filtered.append([None for _ in range(nLevels)])

        self.alpha = alpha
        self.fl = fl
        self.fh = fh
        self.samplingRate = samplingRate
        self.low_a = low_a
        self.low_b = low_b
        self.high_a = high_a
        self.high_b = high_b
        self.width = img1.shape[0]
        self.height = img1.shape[1]
        self.lambd = (self.width ** 2 + self.height ** 2) / 3.0
        self.lambda_c = lambda_c
        self.delta = self.lambda_c / 8.0 / (1 + self.alpha)

    def process_frame(self, img2):
        img2 = img_as_float(img2)
        output_channels = []

        for c in range(3):
            py2 = pt.pyramids.LaplacianPyramid(img2[:, :, c])
            py2._build_pyr()
            pyr = list(py2.pyr_coeffs.values())

            for u in range(len(self.pyramids[c])):
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
                currAlpha = lambd / delta / 8.0 - 1
                currAlpha = currAlpha * exaggeration_factor

                if l == len(filtered) - 1 or l == 0:
                    filtered[l] = np.zeros_like(filtered[l])
                elif currAlpha > self.alpha:
                    filtered[l] = self.alpha * filtered[l]
                else:
                    filtered[l] = currAlpha * filtered[l]

                lambd = lambd / 2.0

            output_channel = reconPyr(filtered)
            output_channels.append(output_channel)

        output = np.stack(output_channels, axis=2)
        output = img2 + output
        output = np.clip(output, 0, 1)
        output = img_as_ubyte(output)
        return output


def process_video(input_path, output_path, alpha, lambda_c, fl, fh, fps, progress_callback=None):
    """
    Process a video file with motion magnification.
    progress_callback: callable(current_frame, total_frames)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    ret, img1 = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise ValueError("Failed to read first frame from video.")

    magnifier = Magnify(img1, alpha, lambda_c, fl, fh, fps)
    out.write(img1)  # write first frame as-is

    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_frame = magnifier.process_frame(frame)
        out.write(out_frame)
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, total_frames)

    cap.release()
    out.release()
    return output_path
