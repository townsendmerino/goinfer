#!/usr/bin/env python3
"""The I3 sandbox (docs/tasks/task-integration-harness-2026-10.md, "I3 sandbox, registered 2026-10-13"): run a command
a small model is steering, with permission prompts bypassed, where only the run's own directories are writable and the
only reachable address is goinfer's. Unprivileged bubblewrap: no sudo, no package, no container image.

  - every namespace unshared (user, pid, ipc, uts, cgroup, net): the network is a private loopback and nothing else;
  - the root is a tmpfs remounted read-only; /usr, the Go toolchain and the harness install are bound read-only; the
    host's home directory, its secrets and its other mounts are not there at all;
  - writable: the fixture copy (/work) and a per-run HOME (/home/agent), both host directories made for the run, and
    an in-memory /tmp that vanishes at exit;
  - the one way out is a unix socket bind-mounted from the host; a socat inside listens on 127.0.0.1:<port> and relays
    to it, a socat on the host relays the socket to goinfer's TCP port. Nothing else is bridged;
  - the environment is cleared: no host variable (no API key) reaches the command;
  - --die-with-parent --new-session: killing the runner kills the sandbox.

selftest() attempts the escapes and must be run, and must pass, before any agent starts (fail closed).
"""
import json, os, shutil, signal, socket, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME_IN = "/home/agent"
WORK_IN = "/work"
SOCK_IN = "/net/goinfer.sock"
# Read-only host paths a command needs; anything not listed does not exist inside.
RO_ETC = ["passwd", "group", "nsswitch.conf", "ld.so.cache", "ld.so.conf", "ld.so.conf.d", "localtime", "ssl", "pki", "alternatives", "crypto-policies"]
GO_ROOT = "/usr/local/go"
# ripgrep, which opencode would download on first use (there is no network): the copy Codex ships, read-only, on PATH
TOOLS = os.path.expanduser("~/harnesses/tools/bin")


def start_forwarder(port):
    """Host side: relay a unix socket to the loopback TCP port of goinfer (or of wire_proxy.py in front of it). The socket
    lives in a short /tmp path of its own (a unix socket path is limited to about 100 bytes). Returns (Popen, socket path)."""
    path = os.path.join(tempfile.mkdtemp(prefix="gi-"), "s")
    p = subprocess.Popen(["socat", f"UNIX-LISTEN:{path},fork,mode=0600", f"TCP:127.0.0.1:{port}"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(path):
            return p, path
        time.sleep(0.05)
    p.kill()
    raise RuntimeError("socat did not create " + path)


def argv(sock, work, home, port, ro_extra=(), env=None, cwd=WORK_IN, cmd=(), weaken=()):
    """The bwrap command line. work and home are host directories; sock is the forwarder's socket path."""
    # weaken: planted defects for selftest_mutants only ("rw-etc", "share-net", "rw-root"); never set by a real run.
    a = ["bwrap", "--unshare-all", *(["--share-net"] if "share-net" in weaken else []), "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
         "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64"]
    for e in RO_ETC:
        if os.path.exists("/etc/" + e):
            a += ["--ro-bind", "/etc/" + e, "/etc/" + e]
    if "rw-etc" in weaken:
        a += ["--bind", home, "/etc/ssl"]   # a writable host directory where a read-only one belongs
    if os.path.isdir(GO_ROOT):
        a += ["--ro-bind", GO_ROOT, GO_ROOT]
    for p in ro_extra:
        a += ["--ro-bind", p, p]
    a += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
          "--bind", work, WORK_IN, "--bind", home, HOME_IN,
          "--ro-bind", sock, SOCK_IN, *([] if "rw-root" in weaken else ["--remount-ro", "/"]), "--chdir", cwd]
    e = {"HOME": HOME_IN, "USER": "agent", "LANG": "C.UTF-8", "TERM": "dumb", "TMPDIR": "/tmp",
         "PATH": f"{GO_ROOT}/bin:{TOOLS}:/usr/local/bin:/usr/bin:/bin", "GOTOOLCHAIN": "local", "GOPROXY": "off", "GOFLAGS": "-buildvcs=false",
         "GOCACHE": HOME_IN + "/.cache/go-build", "GOPATH": HOME_IN + "/go", "GIT_CONFIG_NOSYSTEM": "1",
         "GIT_AUTHOR_NAME": "agent", "GIT_AUTHOR_EMAIL": "agent@sandbox", "GIT_COMMITTER_NAME": "agent", "GIT_COMMITTER_EMAIL": "agent@sandbox"}
    e.update(env or {})
    for k, v in e.items():
        a += ["--setenv", k, v]
    # The relay inside: 127.0.0.1:<port> -> the socket. The command is exec'd once the port answers.
    inner = ('socat TCP-LISTEN:%d,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:%s >/dev/null 2>&1 &\n'
             'for i in $(seq 100); do nc -z 127.0.0.1 %d 2>/dev/null && break; sleep 0.05; done\n'
             'ulimit -u 4096; ulimit -f 4194304\n'
             'exec "$@"') % (port, SOCK_IN, port)
    return a + ["--", "/bin/sh", "-c", inner, "sh"] + list(cmd)


def run(sock, work, home, port, cmd, env=None, ro_extra=(), timeout=600, cwd=WORK_IN, stdin=subprocess.DEVNULL, weaken=()):
    """Run cmd in the sandbox. Returns (returncode or "timeout", stdout, stderr, seconds)."""
    t0 = time.time()
    p = subprocess.Popen(argv(sock, work, home, port, ro_extra, env, cwd, cmd, weaken), stdin=stdin, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True)
    try:
        so, se = p.communicate(timeout=timeout)
        return p.returncode, so, se, time.time() - t0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        so, se = p.communicate()
        return "timeout", so or "", se or "", time.time() - t0


PROBE = r'''
import json, os, socket, sys, urllib.request
port = int(sys.argv[1]); canary = int(sys.argv[2]); lan = sys.argv[3]; hostpid = int(sys.argv[4]); symlink_target = sys.argv[5]
r = {}
def attempt(name, fn):
    try:
        r[name] = ["OK", str(fn())[:80]]
    except Exception as e:
        r[name] = ["FAIL", type(e).__name__ + ": " + str(e)[:80]]
def write(p):
    def f():
        open(p, "w").write("x"); return p
    return f
def tcp(host, prt):
    def f():
        s = socket.create_connection((host, prt), timeout=3); s.close(); return "connected"
    return f
os.makedirs("/work/sub", exist_ok=True)
for p in ["/usr/x", "/etc/x", "/etc/ssl/x", "/usr/local/go/x", "/x", "/home/francis/x", "/bin/x", "/proc/sys/kernel/hostname", "/net/x", "/root/x", "/work/sub/../../x"]:
    attempt("write " + p, write(p))
os.symlink(symlink_target, "/work/link", target_is_directory=False) if not os.path.lexists("/work/link") else None
attempt("write through a symlink in /work to " + symlink_target, write("/work/link"))
os.symlink("/usr/pwned", "/work/link2") if not os.path.lexists("/work/link2") else None
attempt("write through a symlink in /work to /usr/pwned", write("/work/link2"))
attempt("write /work/new.txt", write("/work/new.txt"))
attempt("write /home/agent/new.txt", write("/home/agent/new.txt"))
attempt("write /tmp/new.txt", write("/tmp/new.txt"))
attempt("read /home/francis", lambda: os.listdir("/home/francis"))
attempt("read /etc/shadow", lambda: open("/etc/shadow").read())
attempt("read the host's ~/.ssh", lambda: os.listdir("/home/francis/.ssh"))
attempt("list /run", lambda: os.listdir("/run"))
# network
attempt("tcp 1.1.1.1:443", tcp("1.1.1.1", 443))
attempt("tcp 8.8.8.8:53", tcp("8.8.8.8", 53))
attempt("dns example.com", lambda: socket.getaddrinfo("example.com", 443))
attempt("tcp host loopback canary :%d" % canary, tcp("127.0.0.1", canary))
attempt("tcp host's LAN address %s:%d" % (lan, canary), tcp(lan, canary))
attempt("tcp 127.0.0.1:22", tcp("127.0.0.1", 22))
attempt("tcp 127.0.0.1:%d (a port beside goinfer's)" % (port + 1), tcp("127.0.0.1", port + 1))
attempt("tcp 169.254.169.254:80 (metadata)", tcp("169.254.169.254", 80))
attempt("unix /var/run/docker.sock", lambda: socket.socket(socket.AF_UNIX).connect("/var/run/docker.sock"))
attempt("unix /run/user/1000/bus", lambda: socket.socket(socket.AF_UNIX).connect("/run/user/1000/bus"))
def goinfer():
    d = json.load(urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=5))
    return [m["id"] for m in d["data"]]
attempt("goinfer GET /v1/models", goinfer)
# processes and environment
attempt("signal the host's pid %d" % hostpid, lambda: os.kill(hostpid, 0))
r["pid of this process"] = ["INFO", os.getpid()]
r["environment keys"] = ["INFO", sorted(os.environ)]
print(json.dumps(r))
'''

MUST_WORK = ["write /work/new.txt", "write /home/agent/new.txt", "write /tmp/new.txt", "goinfer GET /v1/models"]


def lan_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        return s.getsockname()[0]
    finally:
        s.close()


def selftest(port, verbose=True, weaken=()):
    """Attempt the escapes from inside; return (ok, report). A host canary listener stands for 'another service on this
    machine'; a pid of a host process stands for 'something to signal'. Afterwards the host is checked for stray files."""
    rundir = tempfile.mkdtemp(prefix="sbx-selftest-")
    work, home = os.path.join(rundir, "work"), os.path.join(rundir, "home")
    os.makedirs(work); os.makedirs(home)
    canary = socket.socket(); canary.bind(("0.0.0.0", 0)); canary.listen(5); cport = canary.getsockname()[1]
    stop = []
    def accept():
        canary.settimeout(0.2)
        while not stop:
            try:
                c, _ = canary.accept(); c.close(); stop.append("CANARY WAS REACHED")
            except OSError:
                pass
    threading.Thread(target=accept, daemon=True).start()
    fw, sock = start_forwarder(port)
    sentinel = tempfile.mkdtemp(prefix="sbx-target-")   # a host directory a symlink in the fixture points into
    target = os.path.join(sentinel, "pwned")
    try:
        rc, so, se, _ = run(sock, work, home, port, ["python3", "-c", PROBE, str(port), str(cport), lan_address(), str(os.getpid()), target], timeout=120, weaken=weaken)
    finally:
        fw.terminate()
    reached = "CANARY WAS REACHED" in stop
    stop.append("done")
    res = json.loads(so.strip().splitlines()[-1]) if so.strip() else {}
    problems = []
    if rc != 0 or not res:
        problems.append(f"the probe did not run: rc {rc} {se[-300:]}")
    for k, v in res.items():            # every attempt but the four allowed ones must fail
        if v[0] in ("OK",) and k not in MUST_WORK:
            problems.append(f"ESCAPE: '{k}' -> {v}")
    if len([k for k in res if res[k][0] == "FAIL"]) < 25:
        problems.append("the probe attempted fewer escapes than it should have")
    for k in MUST_WORK:
        if res.get(k, ["?"])[0] != "OK":
            problems.append(f"the allowed path does not work: '{k}' -> {res.get(k)}")
    if reached:
        problems.append("ESCAPE: the host canary listener was reached from inside")
    for host_path in ["/usr/x", "/etc/x", "/etc/ssl/x", "/usr/local/go/x", "/x", target, "/home/francis/x", os.path.expanduser("~/harnesses/x")]:
        if os.path.exists(host_path):
            problems.append(f"ESCAPE: {host_path} exists on the host")
    leaked = sorted(set(os.listdir(work)) - {"sub", "link", "link2", "new.txt"})
    if leaked:
        problems.append(f"unexpected files in the fixture copy: {leaked}")
    if any(k for k in res.get("environment keys", ["", []])[1] if k not in ("HOME", "USER", "LANG", "TERM", "TMPDIR", "PATH", "GOTOOLCHAIN", "GOPROXY", "GOFLAGS", "GOCACHE", "GOPATH",
            "GIT_CONFIG_NOSYSTEM", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "PWD", "SHLVL", "_", "OLDPWD")):
        problems.append(f"host environment leaked: {res['environment keys']}")
    shutil.rmtree(rundir, ignore_errors=True); shutil.rmtree(sentinel, ignore_errors=True)
    lines = [f"{'ok  ' if (res[k][0] == 'OK') == (k in MUST_WORK) else 'BAD '} {k}: {res[k]}" for k in sorted(res) if res[k][0] != "INFO"]
    report = "\n".join(lines) + f"\nenvironment inside: {res.get('environment keys', [None, None])[1]}\npid inside: {res.get('pid of this process', [None, None])[1]}\n"
    if verbose:
        print(report)
    return not problems, report + ("\n".join(problems) if problems else "SANDBOX SELFTEST PASSED")


def selftest_mutants(port):
    """The gate must be able to go red: each planted weakness has to be caught by selftest. Returns (ok, report)."""
    out, ok = [], True
    for w in ("rw-etc", "share-net", "rw-root"):
        passed, rep = selftest(port, verbose=False, weaken=(w,))
        caught = not passed
        ok = ok and caught
        esc = [l for l in rep.splitlines() if l.startswith("ESCAPE")]
        out.append(f"planted {w}: {'caught' if caught else 'NOT CAUGHT'} ({len(esc)} escapes named, e.g. {esc[0] if esc else '-'})")
    return ok, "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "mutants":
        ok, rep = selftest_mutants(int(sys.argv[1]))
        print(rep)
        sys.exit(0 if ok else 1)
    ok, rep = selftest(int(sys.argv[1]) if len(sys.argv) > 1 else 18090)
    print(rep.splitlines()[-1] if ok else rep)
    sys.exit(0 if ok else 1)
