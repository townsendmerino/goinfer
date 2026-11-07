#!/usr/bin/env python3
"""Attribution for G2's gemma3-1b cell at depth 128 (0.942 against f8632013): the same cell, goinfer against goinfer, main 4db6cd19
(KV5-1 to KV5-5, no KV5-6) as the old arm and 0b2c9bd0 as the new, A B B A, bench_peer.py from 0b2c9bd0's tree. Context, not a gate:
it says whether KV5-6 is where the cost is. Run under the exclusive timing lock, on AC."""
import os, subprocess, sys
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
OUT = f'{D}/g2-day'
env = dict(os.environ, BENCH_DEPTH_BACKEND='metal', BENCH_ENGINES='goinfer,goinfer_old', BENCH_BACKENDS='metal', BENCH_REQUIRE_BACKEND='1',
           BENCH_SERVE_LOG_DIR=f'{OUT}/serve-logs', BENCH_LOAD_TIMEOUT='420', GOINFER_SERVE_METAL=f'{D}/serve-new',
           GOINFER_SERVE_METAL_OLD=f'{D}/serve-main', BENCH_SWAPOUT_VOID_MB='0', BENCH_SWAP_VOID_STOP='3', BENCH_IDLE_WAIT='300',
           BENCH_ABBA='1', BENCH_MODELS='gemma3-1b', BENCH_PHASES='A')
env.pop('BENCH_SWAPIN_VOID_MB', None)
with open(f'{OUT}/gemma-ab-main.log', 'w') as f:
    rc = subprocess.run(['python3', '-u', 'bench_peer.py', f'{OUT}/gemma-ab-main.json'], cwd=f'{D}/new-0b2c9bd0/scripts', env=env,
                        stdout=f, stderr=subprocess.STDOUT).returncode
print('rc', rc)
sys.exit(rc)
