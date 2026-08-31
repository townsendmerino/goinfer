"""G34: does BLOCK VERIFY help the expert-DMA term? Replay of the G33 routing trace.

Reuses G33's validated cache model — scripts/g33_replay.py reproduces the measured
76.1% hit rate on this trace, which is the gate that makes any number below meaningful.

Serial decode issues each token's expert set at each layer separately. A block verify of
width K runs the target over K draft positions in ONE forward pass, so at layer L all K
positions' routing is COMPUTED together and the UNION is requested once. That is not a
prediction — it is already-decided work batched — which is why G33's kill of routing-
PREDICTION prefetch does not rule it out.

Only alpha of the K verified tokens are accepted, so the economics are:

    per accepted token, block = misses_block / ((N/K) * alpha)
    per accepted token, serial= misses_serial / N
    pays iff  misses_block / misses_serial  <  alpha / K
"""
import json, collections, sys

t = json.load(open('docs/measurements/g33-routing-trace.json'))
dec, SLOTS = t['decisions'], t['slots']
layers = sorted({d['layer'] for d in dec})
L = len(layers)
steps = len(dec) // L  # positions per layer

# regroup into [step][layer] -> idx
grid = [[None]*L for _ in range(steps)]
for i, d in enumerate(dec):
    grid[i // L][d['layer']] = d['idx']

def replay(blocks):
    """blocks: list of (layer, set_of_experts) requests in issue order."""
    lru = collections.defaultdict(collections.OrderedDict)
    hits = misses = 0
    for lay, want in blocks:
        c = lru[lay]
        for e in want:
            if e in c:
                c.move_to_end(e); hits += 1
            else:
                misses += 1; c[e] = None
                if len(c) > SLOTS: c.popitem(last=False)
    return hits, misses

# serial baseline: K=1
serial = [(l, grid[s][l]) for s in range(steps) for l in range(L)]
h0, m0 = replay(serial)
print(f"trace: {steps} positions x {L} layers, topK={t['topK']}, slots={SLOTS}")
print(f"SERIAL   K=1   misses {m0}   hit {h0/(h0+m0)*100:.1f}%   {m0/steps:.1f} misses/token\n")
print(f"{'K':>3} {'misses':>8} {'vs serial':>10} {'misses/verified':>16} {'alpha needed to break even':>28}")
for K in (2,3,4,6,8):
    blocks=[]
    for s0 in range(0, steps, K):
        grp = range(s0, min(s0+K, steps))
        for l in range(L):
            u = set()
            for s in grp: u |= set(grid[s][l])
            blocks.append((l, sorted(u)))
    h, m = replay(blocks)
    ratio = m/m0
    need = K*ratio           # alpha must EXCEED this
    print(f"{K:>3} {m:>8} {ratio:>9.3f}x {m/steps:>15.1f} {need:>27.2f}")
