#!/usr/bin/env python3
"""G-S14c4's data (docs/tasks/task-multimodal-support-2026-10.md): the 73 clips of hf-internal-testing/librispeech_asr_dummy (LibriSpeech clean validation, CC BY 4.0) as 16 kHz mono 16-bit WAV
plus refs.json {id: transcript}, and provenance.json (tool versions, durations, word counts). Needs pyarrow and huggingface_hub (a data-only venv: ~/goinfer-bench/venv-data).
Run: ~/goinfer-bench/venv-data/bin/python scripts/s14c4_data.py <out dir>"""
import json, os, subprocess, sys, wave
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
t = pq.read_table(hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
refs, dur, words = {}, 0.0, 0
for r in t:
    cid, src = r["id"], f"{out}/{r['id']}.flac"
    open(src, "wb").write(r["audio"]["bytes"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", f"{out}/{cid}.wav"], check=True)
    w = wave.open(f"{out}/{cid}.wav"); dur += w.getnframes() / 16000; refs[cid] = r["text"]; words += len(r["text"].split())
json.dump(refs, open(f"{out}/refs.json", "w"), indent=0)
ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
json.dump({"clips": len(refs), "seconds": round(dur, 1), "words": words, "ffmpeg": ff, "source": "hf-internal-testing/librispeech_asr_dummy clean/validation", "license": "CC BY 4.0 (LibriSpeech, Panayotov et al. 2015)"}, open(f"{out}/provenance.json", "w"), indent=1)
print(len(refs), "clips,", round(dur, 1), "s,", words, "words")
