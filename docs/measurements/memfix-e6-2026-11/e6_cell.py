"""E6 (memory audit F7): the served process's own peak host memory against the host guard's CUDA expert-cache build term.
For each model: start serve (direct .gguf load, CUDA, int4, C' expert cache), sample /proc/<pid>/status every 0.2 s
(VmHWM, VmRSS, RssAnon, RssFile, RssShmem, VmSwap) until it has answered one short request, then stop it. System swap and
MemAvailable are context only. Run under the exclusive timing lock. Usage: python3 -I e6_cell.py <serve> <outdir> <model>..."""
import json, os, subprocess, sys, time, urllib.request
serve, out = sys.argv[1], sys.argv[2]
KIB = 1024
def status(pid):
    d = {}
    try:
        for l in open(f'/proc/{pid}/status'):
            k, _, v = l.partition(':')
            if k in ('VmHWM', 'VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap'):
                d[k] = int(v.split()[0]) * KIB
    except FileNotFoundError:
        return None
    return d
def meminfo():
    d = {}
    for l in open('/proc/meminfo'):
        k, _, v = l.partition(':')
        d[k] = int(v.split()[0]) * KIB
    return d
def ready(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2) as r:
            return r.status == 200
    except Exception:
        return False
def ask(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10) as r:
        mid = json.loads(r.read())['data'][0]['id']
    body = json.dumps({'model': mid, 'messages': [{'role': 'user', 'content': 'Say hello in five words.'}], 'max_tokens': 8, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())['choices'][0]['message']['content']
GB = 1 << 30
for i, model in enumerate(sys.argv[3:]):
    name = os.path.basename(model)
    port = 18130 + i
    mi0 = meminfo()
    load0 = open('/proc/loadavg').read().split()[0]
    log = open(os.path.join(out, f'serve-{name}.log'), 'w')
    t0 = time.time()
    p = subprocess.Popen([serve, '-model', model, '-addr', f'127.0.0.1:{port}', '-backend', 'cuda', '-quant', 'int4', '-moe-cache-experts', '-direct-load'],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    peak = {}
    last = time.time()
    t_ready = answer = None
    while p.poll() is None:
        s = status(p.pid)
        if s:
            for k, v in s.items():
                peak[k] = max(peak.get(k, 0), v)
            peak['hwm_last'] = s.get('VmHWM', 0)
        now = time.time()
        if now - last > 30:
            print(f'[e6] {name} {now-t0:5.0f}s VmRSS {s.get("VmRSS",0)/GB:.2f} GiB VmHWM {s.get("VmHWM",0)/GB:.2f} VmSwap {s.get("VmSwap",0)/GB:.3f}', flush=True)
            last = now
        if t_ready is None and ready(port):
            t_ready = now - t0
            try:
                answer = ask(port)
            except Exception as e:
                answer = f'REQUEST FAILED: {e}'
            break
        if now - t0 > 1800:
            print(f'[e6] {name}: not ready in 30 min, stopping', flush=True)
            break
        time.sleep(0.2)
    s = status(p.pid) or {}
    for k, v in s.items():
        peak[k] = max(peak.get(k, 0), v)
    rc = p.poll()
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
    mi1 = meminfo()
    rec = {'model': model, 'load1_at_start': load0, 'MemAvailable_at_start': mi0['MemAvailable'], 'SystemSwapUsed_at_start': mi0['SwapTotal'] - mi0['SwapFree'],
           'SystemSwapUsed_at_end': mi1['SwapTotal'] - mi1['SwapFree'], 'ready_s': t_ready, 'exited_early_rc': rc, 'answer': answer,
           'peak': peak}
    json.dump(rec, open(os.path.join(out, f'e6-{name}.json'), 'w'), indent=1)
    print(f'[e6] {name}: ready {t_ready}s, VmHWM {peak.get("VmHWM",0)/GB:.2f} GiB, peak VmSwap {peak.get("VmSwap",0)/GB:.3f} GiB, '
          f'peak RssAnon {peak.get("RssAnon",0)/GB:.2f} RssFile {peak.get("RssFile",0)/GB:.2f} RssShmem {peak.get("RssShmem",0)/GB:.2f}; answer {answer!r}', flush=True)
