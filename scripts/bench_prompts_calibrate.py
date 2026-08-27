"""Token-calibrate benchmark prompts and MERGE them into scripts/prompts.json.

A depth axis has to be real token depth, not a word count: the first attempt at this used "itemN"
filler, which tokenizes at ~4 tokens/word, so "depth 2048" was really 9094 tokens and blew the 4096
context window — 400ing both engines identically, which looks like agreement rather than a bug.
Every prompt here is measured against goinfer's own usage.prompt_tokens for the model it will be
used with, because tokenizers differ per family and a prompt calibrated on qwen is not 512 tokens
on phi3.

    python3 scripts/bench_prompts_calibrate.py --serve ./goinfer-serve \
        --model 0.5B=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
        --depth 256 --depth 1024

MERGES by default: existing entries for other models/depths are preserved. The previous version of
this script hardcoded a /tmp scratchpad path for both the serve binary and its output, and called
json.dump on a fresh dict — so it could not run at all once that scratchpad was cleared, and would
have destroyed every calibration it did not itself produce.
"""
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "prompts.json")
PORT = int(os.environ.get("CALIB_PORT", "8099"))


def wait_port(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def prompt_tokens(text):
    payload = {"model": "bench", "stream": False, "max_tokens": 1, "temperature": 0,
               "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)["usage"]["prompt_tokens"]


# " the" is a single common token in every tokenizer here, so repetition gives ~1 token/word and the
# search converges in a couple of steps.
def make(n):
    return "Continue this text. " + " ".join(["the"] * n)


def calibrate(depths, tolerance_pct=1.0):
    out = {}
    for target in depths:
        n, best = target, None
        for _ in range(12):
            got = prompt_tokens(make(n))
            if best is None or abs(got - target) < abs(best[1] - target):
                best = (n, got)
            if abs(got - target) <= max(3, target * tolerance_pct / 100):
                break
            n = max(1, int(n * target / max(got, 1)))
        out[target] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True, help="path to a built goinfer serve binary")
    ap.add_argument("--model", action="append", required=True, metavar="KEY=PATH",
                    help="repeatable; KEY is the prompts.json key prefix")
    ap.add_argument("--depth", action="append", type=int, required=True, help="repeatable")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--quant", default="int4")
    ap.add_argument("--ctx", type=int, default=0,
                    help="pass -ctx to serve; required to calibrate depths past the default cap")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    existing = {}
    if os.path.exists(args.out):
        existing = json.load(open(args.out))
        print(f"merging into {args.out} ({len(existing)} existing prompt(s))")

    added = 0
    for spec in args.model:
        key, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            sys.exit(f"calibrate: no model at {path}")
        # The bench rule: never measure from the archive. Calibration only reads a few tokens, but
        # a path under the archive here means the SAME path reaches the timed run.
        if path.startswith("/srv/models") or path.startswith("/Volumes/"):
            sys.exit(f"calibrate: {path} is under the archive — copy it to ~/models first")
        proc = subprocess.Popen(
            [args.serve, "-model", f"bench={path}", "-backend", args.backend,
             "-addr", f"127.0.0.1:{PORT}", "-quant", args.quant]
            + (["-ctx", str(args.ctx)] if args.ctx else []),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        try:
            if not wait_port(PORT):
                sys.exit(f"calibrate: server for {key} never came up")
            for target, (words, tokens) in calibrate(args.depth).items():
                existing[f"{key}:{target}"] = {"words": words, "tokens": tokens,
                                               "text": make(words)}
                added += 1
                print(f"  {key} depth={target}: {tokens} tokens ({words} words)", flush=True)
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            time.sleep(3)

    json.dump(existing, open(args.out, "w"), indent=1, sort_keys=True)
    print(f"wrote {args.out}: {added} calibrated, {len(existing)} total")


if __name__ == "__main__":
    main()
