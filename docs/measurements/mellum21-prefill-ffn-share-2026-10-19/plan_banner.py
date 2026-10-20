#!/usr/bin/env python3
"""Print the CUDA plan lines a serve start gives one load (the lines before "goinfer serving"), with the free-VRAM and timing figures LEFT IN: the question is
how many expert-cache slots a context leaves, which is the server's own plan and not arithmetic (CLAUDE.md, "A fit check is the server's plan").

  python3 plan_banner.py BINARY MODEL_PATH CTX [extra serve flags...]     (run under the timing lock)
"""
import os, signal, socket, subprocess, sys, time

PORT = 18872


def up(proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            socket.create_connection(("127.0.0.1", PORT), 1).close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    binary, path, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
    log = f"/tmp/plan_banner_{os.getpid()}.log"
    argv = [binary, "-model", f"bench={path}", "-backend", "cuda", "-require-backend", "-addr", f"127.0.0.1:{PORT}", "-ctx", ctx] + sys.argv[4:]
    with open(log, "w") as sink:
        proc = subprocess.Popen(argv, stdout=sink, stderr=sink, preexec_fn=os.setsid)
        ok = up(proc, 900)
        time.sleep(1)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        time.sleep(2)
    print("argv:", " ".join(argv))
    for ln in open(log).read().splitlines():
        if ln.startswith("goinfer serving"):
            break
        if ln.startswith(("swap guard", "admin socket")):
            continue
        print(ln[:400])
    print("UP" if ok else "VOID: the server did not come up")
    os.unlink(log)
    sys.exit(0 if ok else 1)


main()
