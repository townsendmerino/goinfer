#!/usr/bin/env python3
"""G-S5e's inputs (docs/tasks/task-multimodal-support-2026-10.md, S5's follow-up), made offline from the LibriSpeech clip
(testdata/speech/librispeech-1272-128104-0000.wav, 16 kHz mono, CC BY 4.0) by tools independent of goinfer:

  -44k1.wav           (i)  44.1 kHz mono, 16-bit: ffmpeg's default resampler, -ar 44100 -ac 1 -c:a pcm_s16le
  -48k-stereo.wav     (ii) 48 kHz stereo, 16-bit: ffmpeg to 48 kHz mono, then L = s + d, R = s - d with d = 0.3 x s
                           time-reversed, so the per-sample mean is s (up to rounding) and the left channel alone is not
  -44k1.ref16k.f32    (iii) (i) back to 16 kHz with scipy.signal.resample_poly (Kaiser window, defaults), float32 LE
  -48k-stereo.ref16k.f32    (ii) downmixed by the mean, then the same

Run from the repo root: python3 scripts/make_resample_clips.py   (needs ffmpeg, numpy, scipy)
"""
import json
import os
import subprocess
import wave

import numpy as np
import scipy
from scipy.signal import resample_poly

D = "testdata/speech"
SRC = os.path.join(D, "librispeech-1272-128104-0000.wav")
BASE = os.path.join(D, "librispeech-1272-128104-0000")


def read16(path):
    with wave.open(path) as w:
        assert w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
        return x.reshape(-1, w.getnchannels()), w.getframerate()


def write16(path, x, rate):
    q = np.clip(np.round(x * 32768), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(q.tobytes())


def ffmpeg(rate, out):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", SRC, "-ar", str(rate), "-ac", "1",
                    "-c:a", "pcm_s16le", out], check=True)


def main():
    ffv = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    ffmpeg(44100, BASE + "-44k1.wav")
    tmp = BASE + "-48k-mono.tmp.wav"
    ffmpeg(48000, tmp)
    s, _ = read16(tmp)
    os.remove(tmp)
    s = s[:, 0]
    d = 0.3 * s[::-1]
    lr = np.stack([s + d, s - d], axis=1)
    peak = np.abs(lr).max()
    assert peak < 1.0, f"stereo would clip (peak {peak})"
    write16(BASE + "-48k-stereo.wav", lr, 48000)
    x44, r44 = read16(BASE + "-44k1.wav")
    assert r44 == 44100
    resample_poly(x44[:, 0], 160, 441).astype("<f4").tofile(BASE + "-44k1.ref16k.f32")
    x48, r48 = read16(BASE + "-48k-stereo.wav")
    assert r48 == 48000 and x48.shape[1] == 2
    resample_poly(x48.mean(axis=1), 1, 3).astype("<f4").tofile(BASE + "-48k-stereo.ref16k.f32")
    json.dump({"ffmpeg": ffv, "numpy": np.__version__, "scipy": scipy.__version__, "stereo_peak": float(peak)},
              open(BASE + "-resample.provenance.json", "w"), indent=1)
    print(ffv, "| numpy", np.__version__, "| scipy", scipy.__version__, "| stereo peak", round(float(peak), 4))


if __name__ == "__main__":
    main()
