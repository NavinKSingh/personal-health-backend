"""
rPPG Processor — Remote Photoplethysmography
─────────────────────────────────────────────────────────────────────────────
Measures heart rate from camera frames by tracking subtle skin color changes.

Algorithm (CHROM method, De Haan & Jeanne 2013):
  1. Face detection using OpenCV Haar cascade
  2. Extract mean R, G, B values from face ROI each frame
  3. Collect 10-second sliding window (300 frames at 30fps, or variable)
  4. Apply CHROM algorithm: orthogonal chrominance signals Xs, Ys
  5. Bandpass filter 0.67–3.0 Hz (40–180 BPM)
  6. FFT → dominant frequency → BPM

Why CHROM over plain green-channel:
  - Cancels illuminant-dependent noise (RGB fluctuations from lights)
  - More robust to skin tone differences
  - Less sensitive to subtle camera shake

References:
  - De Haan & Jeanne, IEEE Trans Bio-Med Eng, 2013
  - Wang et al., IEEE Trans Bio-Med Eng, 2017 (POS method)
  - rPPG-Toolbox (open source benchmark)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import time
from collections import deque
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from scipy.signal import butter, filtfilt


# ─── Face Detector (Haar cascade — fast, no model download needed) ─────────

_face_detector: Optional[cv2.CascadeClassifier] = None


def _get_face_detector() -> cv2.CascadeClassifier:
    global _face_detector
    if _face_detector is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_detector = cv2.CascadeClassifier(xml)
    return _face_detector


# ─── Bandpass Filter ────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: float, low: float = 0.67, high: float = 3.0) -> np.ndarray:
    """Butterworth bandpass filter. fs = sample rate in Hz."""
    nyq = 0.5 * fs
    low_n  = low  / nyq
    high_n = high / nyq
    # Clamp to valid range
    low_n  = max(0.01, min(low_n,  0.99))
    high_n = max(0.01, min(high_n, 0.99))
    if low_n >= high_n:
        return signal
    b, a = butter(4, [low_n, high_n], btype='band')
    if len(signal) < 15:  # not enough for filtfilt
        return signal
    return filtfilt(b, a, signal)


# ─── CHROM rPPG ─────────────────────────────────────────────────────────────

def _chrom_bvp(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    CHROM method: compute blood volume pulse signal.
    Returns 1D BVP signal array, same length as input.
    """
    r = r.astype(float)
    g = g.astype(float)
    b = b.astype(float)

    # Normalize by mean (remove DC)
    r_n = r / (np.mean(r) + 1e-9)
    g_n = g / (np.mean(g) + 1e-9)
    b_n = b / (np.mean(b) + 1e-9)

    # CHROM chrominance signals
    xs = 3 * r_n - 2 * g_n          # Xs
    ys = 1.5 * r_n + g_n - 1.5 * b_n  # Ys

    # Standardize
    std_xs = np.std(xs) + 1e-9
    std_ys = np.std(ys) + 1e-9
    alpha  = std_xs / std_ys

    bvp = xs - alpha * ys
    return bvp


# ─── RPPGProcessor ──────────────────────────────────────────────────────────

class RPPGProcessor:
    """
    Per-session stateful rPPG processor.

    Usage:
        proc = RPPGProcessor()
        proc.add_frame(base64_jpeg_string, timestamp=time.time())
        result = proc.compute()  # call any time
    """

    WINDOW_SEC     = 10.0     # sliding window length (seconds)
    MIN_FRAMES     = 20       # lowered: 20 frames sufficient for FFT at ~3fps (cuts warmup from 15s to 7s)
    TARGET_FPS     = 10.0     # realistic FPS over WiFi proxy
    BPM_LOW        = 40
    BPM_HIGH       = 180
    FACE_REDETECT  = 5        # re-run Haar every N frames; use cached bbox otherwise

    def __init__(self):
        self._r:  deque = deque()
        self._g:  deque = deque()
        self._b:  deque = deque()
        self._ts: deque = deque()

        self.last_bpm:      float = 0.0
        self.last_hrv:      float = 0.0
        self.last_quality:  str   = "waiting"
        self.last_waveform: list  = []
        self.frames_total:  int   = 0
        self.face_found_count: int = 0

        # Face detection cache — avoids running Haar every frame
        self._cached_face: Optional[tuple] = None   # (x, y, fw, fh)
        self._frames_since_detect: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def add_frame(self, image_b64: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Decode base64 JPEG → detect face → extract R/G/B means → append to window.

        Returns a dict with immediate signal_quality and face detection result.
        """
        t = timestamp or time.time()
        self.frames_total += 1

        try:
            img_bytes = base64.b64decode(image_b64)
            arr       = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        except Exception:
            return {"face_found": False, "signal_quality": "bad_frame", "error": "decode_failed"}

        if frame is None:
            return {"face_found": False, "signal_quality": "bad_frame"}

        # Resize for speed (target ≤320x240 — Haar works fine at this res)
        h, w = frame.shape[:2]
        scale = min(320 / w, 240 / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection — run Haar every FACE_REDETECT frames, cache result otherwise
        self._frames_since_detect += 1
        if self._cached_face is None or self._frames_since_detect >= self.FACE_REDETECT:
            det   = _get_face_detector()
            faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            self._frames_since_detect = 0
            if len(faces) > 0:
                # Pick the largest face
                self._cached_face = tuple(sorted(faces, key=lambda f: f[2] * f[3])[-1])
            else:
                self._cached_face = None
                return {"face_found": False, "signal_quality": "no_face"}

        if self._cached_face is None:
            return {"face_found": False, "signal_quality": "no_face"}

        # Unpack cached face bbox
        x, y, fw, fh = int(self._cached_face[0]), int(self._cached_face[1]), \
                        int(self._cached_face[2]), int(self._cached_face[3])
        self.face_found_count += 1

        # Take forehead ROI (top 45% of face box) — avoids mouth motion
        roi_y2 = y + int(fh * 0.45)
        roi_x1 = x + int(fw * 0.15)
        roi_x2 = x + int(fw * 0.85)
        roi = frame[y: roi_y2, roi_x1: roi_x2]

        if roi.size == 0:
            return {"face_found": True, "signal_quality": "bad_roi"}

        # BGR → per-channel means
        b_mean, g_mean, r_mean = cv2.split(roi.astype(float))
        r_val = float(r_mean.mean())
        g_val = float(g_mean.mean())
        b_val = float(b_mean.mean())

        # Append to window
        self._r.append(r_val)
        self._g.append(g_val)
        self._b.append(b_val)
        self._ts.append(t)

        # Trim window
        self._trim_window()

        quality = self._signal_quality()
        return {"face_found": True, "signal_quality": quality, "r": r_val, "g": g_val, "b": b_val}

    def compute(self) -> Dict[str, Any]:
        """
        Compute BPM from current window.
        Returns: { bpm, hrv_ms, signal_quality, waveform, fps, frames_in_window, status }
        """
        n = len(self._r)
        quality = self._signal_quality()

        if n < self.MIN_FRAMES:
            return {
                "bpm":              self.last_bpm,
                "hrv_ms":           self.last_hrv,
                "signal_quality":   quality,
                "waveform":         self.last_waveform,
                "frames_in_window": n,
                "status":           "warmup",
                "message":          f"Collecting signal… {n}/{self.MIN_FRAMES} frames",
            }

        # Estimate effective sample rate
        ts_arr = np.array(self._ts)
        elapsed = ts_arr[-1] - ts_arr[0]
        fs = (n - 1) / (elapsed + 1e-9)
        fs = float(np.clip(fs, 5.0, 60.0))

        r  = np.array(self._r)
        g  = np.array(self._g)
        b  = np.array(self._b)

        # CHROM BVP
        bvp = _chrom_bvp(r, g, b)

        # Bandpass
        bvp_f = _bandpass(bvp, fs, 0.67, 3.0)

        # FFT
        n_fft = len(bvp_f)
        freqs   = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        power   = np.abs(np.fft.rfft(bvp_f)) ** 2

        # Restrict to BPM range
        mask    = (freqs >= self.BPM_LOW / 60.0) & (freqs <= self.BPM_HIGH / 60.0)
        if mask.sum() == 0:
            bpm = self.last_bpm or 75.0
        else:
            peak_idx = np.argmax(power[mask])
            bpm = float(freqs[mask][peak_idx] * 60.0)

        # SNR-based quality
        peak_power = float(power[mask].max()) if mask.sum() > 0 else 0
        total_power = float(power.sum()) + 1e-9
        snr = peak_power / total_power

        if snr > 0.25:
            quality_str = "excellent"
        elif snr > 0.12:
            quality_str = "good"
        elif snr > 0.05:
            quality_str = "fair"
        else:
            quality_str = "poor"

        # HRV approximation: std of inter-beat intervals estimated from BVP peaks
        hrv_ms = self._estimate_hrv(bvp_f, fs)

        # Waveform: downsample to 50 points for the app
        wf_ds = bvp_f[::max(1, len(bvp_f) // 50)].tolist()
        # Normalize to -1..1
        wf_max = max(abs(v) for v in wf_ds) or 1.0
        wf_norm = [round(v / wf_max, 3) for v in wf_ds]

        self.last_bpm      = round(bpm, 1)
        self.last_hrv      = hrv_ms
        self.last_quality  = quality_str
        self.last_waveform = wf_norm

        return {
            "bpm":              self.last_bpm,
            "hrv_ms":           hrv_ms,
            "signal_quality":   quality_str,
            "waveform":         wf_norm,
            "fps":              round(fs, 1),
            "frames_in_window": n,
            "snr":              round(snr, 4),
            "status":           "ok",
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _trim_window(self) -> None:
        """Keep only WINDOW_SEC of data, but guarantee we keep at least MIN_FRAMES."""
        while len(self._ts) > self.MIN_FRAMES:
            if self._ts[-1] - self._ts[0] <= self.WINDOW_SEC:
                break
            self._r.popleft()
            self._g.popleft()
            self._b.popleft()
            self._ts.popleft()

    def _signal_quality(self) -> str:
        n = len(self._r)
        if n < 10:
            return "waiting"
        if n < self.MIN_FRAMES:
            return "warmup"
        return "measuring"

    def _estimate_hrv(self, bvp: np.ndarray, fs: float) -> float:
        """
        Estimate RMSSD (root mean square of successive differences) from BVP peaks.
        Returns HRV in ms.
        """
        try:
            from scipy.signal import find_peaks
            # Find systolic peaks
            min_dist = int(fs * 60.0 / 180)  # max 180 BPM
            peaks, _ = find_peaks(bvp, distance=min_dist, prominence=0.05)
            if len(peaks) < 3:
                return 0.0
            ibi_s = np.diff(peaks) / fs  # inter-beat intervals in seconds
            ibi_ms = ibi_s * 1000.0
            rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
            return round(min(rmssd, 200.0), 1)
        except Exception:
            return 0.0
