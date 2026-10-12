#!/usr/bin/env python3
"""The same image turn through two serve binaries under --backend webgpu: are the greedy replies equal?
(docs/tasks/task-multimodal-support-2026-10.md, "S12.4b-4a".)

    served_image_reply.py --new BIN --old BIN --image PNG --log-dir DIR -- <serve args common to both, e.g. --model DIR>

Each binary is started once and sent one request: the image's own bytes (not a fresh copy, so both see the same
pixels), "Describe this image.", temperature 0, --ngen tokens. Prints both replies and each server's lines about
where the image prefill ran. Exit 0 when the replies are equal, 1 when they differ.
"""
import argparse, base64, json, os, signal, subprocess, sys, time, urllib.request

PORT = 18656


def run(binary, args, image, ngen, log_path):
    argv = [binary, *args, "--backend", "webgpu", "--addr", f"127.0.0.1:{PORT}"]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline, model = time.time() + 900, None
        while model is None:
            if proc.poll() is not None or time.time() > deadline:
                sys.exit(f"{binary} did not come up; see {log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
                    model = json.load(r)["data"][0]["id"]
            except Exception:
                time.sleep(0.5)
        url = "data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode()
        body = json.dumps({"model": model, "max_tokens": ngen, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe this image."}]}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            reply = json.load(r)["choices"][0]["message"]["content"]
        return reply, time.time() - t0
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ngen", type=int, default=32)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("serve_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = [x for x in a.serve_args if x != "--"]
    os.makedirs(a.log_dir, exist_ok=True)
    out = {}
    for arm, binary in (("old", a.old), ("new", a.new)):
        log = os.path.join(a.log_dir, f"reply-{arm}.log")
        out[arm], secs = run(binary, args, a.image, a.ngen, log)
        print(f"{arm} ({os.path.basename(binary)}, {secs:.1f} s): {out[arm]!r}")
        for line in open(log):
            if "prefill" in line and ("vision" in line or "image" in line or "declin" in line):
                print(f"    {arm} log: {line.strip()[:200]}")
    same = out["old"] == out["new"]
    print("REPLIES EQUAL" if same else "REPLIES DIFFER")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
