#!/usr/bin/env python3
"""G-S14i3's client (docs/tasks/task-multimodal-support-2026-10.md, "G-S14i"): POST /v1/audio/transcriptions of a served Whisper model, compared with transformers' references.
Usage: s14i_served.py <port> <libri.wav> <ts-real golden.json (G-S14g3's)> <policy-real golden.json (G-S14h2's)> <long70.in.f32 from the policy reference>
The 5.9 s clip's verbose_json segments must equal the g3 reference's (one window), the 70 s composite's the h2 reference's long70 case (13 segments; OpenAI's policy), the language "english"; srt, vtt and text of the
same request must be consistent with the segments; a repeat of a request returns the same result."""
import json, struct, sys, urllib.request, uuid, numpy as np
port, wavp, g3p, h2p, f32p = sys.argv[1:6]
g3 = json.load(open(g3p))["cases"]["libri"]; h2 = json.load(open(h2p))["cases"]["long70"]
def wav_bytes(x16):
    n = len(x16) * 2
    return b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", n) + x16.astype("<i2").tobytes()
long70 = np.fromfile(f32p, dtype="<f4")
q = np.round(long70 * 32768.0); assert np.array_equal(q / 32768.0, long70.astype(np.float64)), "the composite is not 16-bit PCM exactly"
files = {"libri": (open(wavp, "rb").read(), g3), "long70": (wav_bytes(q.astype(np.int16)), h2)}
def post(wav, **fields):
    b = uuid.uuid4().hex; body = b""
    for k, v in fields.items(): body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + wav + f"\r\n--{b}--\r\n".encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={b}"}), timeout=1800)
    return r.read().decode()
def clock(s, sep):
    ms = int(s * 1000 + 0.5); return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"
bad = 0
for name, (wav, ref) in files.items():
    v = json.loads(post(wav, response_format="verbose_json")); segs = v["segments"]
    ok = (v["language"] == "english" and len(segs) == len(ref["segments"]) and all(abs(s["start"] - r["start"]) < 1e-9 and abs(s["end"] - r["end"]) < 1e-9 and s["tokens"] == r["tokens"] for s, r in zip(segs, ref["segments"]))
          and v["text"] == ref["text"].strip())
    srt, vtt, txt = post(wav, response_format="srt"), post(wav, response_format="vtt"), post(wav, response_format="text")
    exp_srt = "".join(f"{i + 1}\n{clock(s['start'], ',')} --> {clock(s['end'], ',')}\n{s['text'].replace('-->', '->')}\n\n" for i, s in enumerate(segs))
    exp_vtt = "WEBVTT\n\n" + "".join(f"{clock(s['start'], '.')} --> {clock(s['end'], '.')}\n{s['text'].replace('-->', '->')}\n\n" for s in segs)
    cons = srt == exp_srt and vtt == exp_vtt and txt == v["text"] + "\n"
    again = json.loads(post(wav, response_format="verbose_json"))
    rep = [s["tokens"] for s in again["segments"]] == [s["tokens"] for s in segs]
    print(f"{name}: {len(segs)} segments (reference {len(ref['segments'])}); equal to transformers {ok}; srt/vtt/text consistent {cons}; repeat identical {rep}; duration {v['duration']:.2f} s; text {v['text'][:70]!r}")
    bad += (not ok) + (not cons) + (not rep)
print("G-S14i3:", "PASS" if bad == 0 else f"FAIL ({bad} checks)")
sys.exit(1 if bad else 0)
