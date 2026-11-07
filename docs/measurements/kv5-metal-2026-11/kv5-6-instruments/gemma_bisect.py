#!/usr/bin/env python3
"""Attribution for G2's gemma3-1b cell at depth 128, which FAILs against f8632013 (0.942 on 2ffc982e, 0.952 on bf5e0a59): context, not a
gate. bench_peer.py's phase A on gemma3-1b at depth 128 (phase A) and 3,900 (phase B), goinfer against goinfer_old, BENCH_ABBA quads, from e2762eb0's tree.
  prep C ...                 an archive of each commit and its serve binary, serve-bis-<c>; run under the shared timing lock
  ab NAME OLD NEW [QUADS]    OLD and NEW are binary paths or commits prepared above; writes gemma-bisect/NAME.json and .log; run under
                             the exclusive timing lock, on AC
  sum [NAME ...]             new/old median over every run of each arm, each quad's two halves, swap-outs, RSS"""
import glob, io, json, os, statistics as S, subprocess, sys, tarfile

D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
WT = '/Users/francistownsend-merino/tmcode/goinfer/.claude/worktrees/op4m-c-2026-10-31'
SCRIPTS = f'{D}/new-e2762eb0/scripts'
OUT = f'{D}/gemma-bisect'
KNOWN = {'f8632013': os.path.expanduser('~/goinfer-work/kv5-2-g2/serve-old'), 'e2762eb0': f'{D}/serve-final2',
         '8b7e9941': f'{D}/serve-main-final2'}


def binary(x):
    if os.path.sep in x:
        return x
    return KNOWN.get(x) or f'{D}/serve-bis-{x}'


def prep(commits):
    for c in commits:
        tree, out = f'{D}/bis-{c}', f'{D}/serve-bis-{c}'
        if os.path.exists(out):
            print(c, 'built before')
            continue
        if not os.path.exists(tree):
            data = subprocess.run(['git', '-C', WT, 'archive', c], capture_output=True, check=True).stdout
            tarfile.open(fileobj=io.BytesIO(data)).extractall(tree, filter='data')
            if os.path.exists(f'{tree}/go.work.example') and not os.path.exists(f'{tree}/go.work'):
                with open(f'{tree}/go.work.example') as s, open(f'{tree}/go.work', 'w') as d:
                    d.write(s.read())
        mod = f'{tree}/backend/metal' if os.path.exists(f'{tree}/backend/metal/cmd/serve') else f'{tree}/metal'
        env = dict(os.environ, GOMAXPROCS='4')
        if os.path.exists(f'{tree}/go.work'):
            env['GOWORK'] = f'{tree}/go.work'
        r = subprocess.run(['go', '-C', mod, 'build', '-p', '2', '-o', out, './cmd/serve'], capture_output=True, text=True, env=env)
        print(c, mod[len(tree) + 1:], 'rc', r.returncode, r.stderr[-400:], flush=True)
        if r.returncode:
            sys.exit(1)


def ab(name, old, new, quads):
    os.makedirs(f'{OUT}/serve-logs', exist_ok=True)
    out = f'{OUT}/{name}.json'
    if os.path.exists(out):
        print(name, 'done before')
        return
    batt = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout.splitlines()[0]
    if 'AC Power' not in batt:
        sys.exit(f'not on AC power: {batt}')
    env = dict(os.environ, BENCH_DEPTH_BACKEND='metal', BENCH_ENGINES='goinfer,goinfer_old', BENCH_BACKENDS='metal', BENCH_REQUIRE_BACKEND='1',
               BENCH_SERVE_LOG_DIR=f'{OUT}/serve-logs', BENCH_LOAD_TIMEOUT='420', GOINFER_SERVE_METAL=binary(new),
               GOINFER_SERVE_METAL_OLD=binary(old), BENCH_SWAPOUT_VOID_MB='0', BENCH_SWAP_VOID_STOP='3', BENCH_IDLE_WAIT='300',
               BENCH_ABBA=str(quads), BENCH_MODELS='gemma3-1b', BENCH_DEPTHS='3900')
    env.pop('BENCH_SWAPIN_VOID_MB', None)
    for b in (env['GOINFER_SERVE_METAL'], env['GOINFER_SERVE_METAL_OLD']):
        if not os.path.exists(b):
            sys.exit(f'missing {b}')
    print(f'{name}: old {old} ({binary(old)}), new {new} ({binary(new)}), {quads} quads; {batt}', flush=True)
    with open(f'{OUT}/{name}.log', 'w') as f:
        rc = subprocess.run(['python3', '-u', 'bench_peer.py', out], cwd=SCRIPTS, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(name, 'rc', rc, flush=True)
    if rc:
        if os.path.exists(out):
            os.rename(out, out + '.failed')
        sys.exit(75 if any(l.startswith('REFUSED') for l in open(f'{OUT}/{name}.log')) else 1)


def summary(names):
    for name in names or sorted(os.path.basename(f)[:-5] for f in glob.glob(f'{OUT}/*.json')):
      for dep in (128, 3900):
        d = [c for c in json.load(open(f'{OUT}/{name}.json')) if c.get('engine') and c.get('depth') == dep]
        if not d:
            continue
        new = [x for c in d if c['engine'] == 'goinfer' for x in c['runs']]
        old = [x for c in d if c['engine'] == 'goinfer_old' for x in c['runs']]
        halves = []
        for b in sorted({c.get('block') for c in d}):
            bn = [x for c in d if c['engine'] == 'goinfer' and c.get('block') == b for x in c['runs']]
            bo = [x for c in d if c['engine'] == 'goinfer_old' and c.get('block') == b for x in c['runs']]
            if bn and bo:
                halves.append(round(S.median(bn) / S.median(bo), 3))
        so = max(c['counts']['swap']['swapout_mb'] for c in d)
        print(f'{name:20s} {dep:5d} new {S.median(new):6.1f}  old {S.median(old):6.1f} tok/s  new/old {S.median(new) / S.median(old):.3f}  '
              f'halves {halves}  runs new {len(new)} old {len(old)}  swap-outs {so} MB')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(sys.argv[2:])
    elif cmd == 'ab':
        ab(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    elif cmd == 'sum':
        summary(sys.argv[2:])
    else:
        sys.exit(f'no command {cmd}')
