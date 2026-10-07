#!/usr/bin/env python3
"""swapwatch.py PID OUT.csv [vmswap_limit_mb=512] [min_avail_mb=3000]
Samples every 0.5 s: the server's own VmSwap (attributable, unlike the system-wide counter), system swap-used, pswpin/pswpout, MemAvailable.
Stops the server (SIGTERM, then SIGKILL) if ITS swap passes the limit or MemAvailable falls under the floor — the first-hour protocol's
amended rule (docs/tasks/task-first-hour.md rule 2): the trigger is pageouts or growth past the guard's threshold, not any swap-used rise."""
import os, sys, time, signal
pid=int(sys.argv[1]); out=sys.argv[2]; lim=int(sys.argv[3]) if len(sys.argv)>3 else 512; floor=int(sys.argv[4]) if len(sys.argv)>4 else 3000
def kb(path,key):
    try:
        for l in open(path):
            if l.startswith(key): return int(l.split()[1])
    except Exception: return None
def swap_used():
    t=f=0
    for l in open('/proc/swaps').read().splitlines()[1:]:
        p=l.split(); t+=int(p[2]); f+=int(p[3])
    return f//1  # KB used
def vm(k):
    for l in open('/proc/vmstat'):
        if l.startswith(k+' '): return int(l.split()[1])
base_out=vm('pswpout'); base_in=vm('pswpin'); base_swap=swap_used(); t0=time.time(); peak=0; w=open(out,'w')
w.write('t_s,server_vmswap_mb,sys_swap_used_mb,d_pswpout_pages,d_pswpin_pages,mem_avail_mb,server_rss_mb\n')
while True:
    if not os.path.exists(f'/proc/{pid}'): w.write('# server exited\n'); break
    vs=(kb(f'/proc/{pid}/status','VmSwap:') or 0)//1024; rss=(kb(f'/proc/{pid}/status','VmRSS:') or 0)//1024
    av=(kb('/proc/meminfo','MemAvailable:') or 0)//1024; su=(swap_used()-base_swap)//1024
    peak=max(peak,vs)
    w.write(f'{time.time()-t0:.1f},{vs},{su},{vm("pswpout")-base_out},{vm("pswpin")-base_in},{av},{rss}\n'); w.flush()
    if vs>lim or av<floor:
        w.write(f'# STOP: server VmSwap {vs} MB (limit {lim}) / MemAvailable {av} MB (floor {floor})\n'); w.flush()
        os.kill(pid,signal.SIGTERM); time.sleep(3)
        if os.path.exists(f'/proc/{pid}'): os.kill(pid,signal.SIGKILL)
        break
    time.sleep(0.5)
w.write(f'# peak server VmSwap {peak} MB\n'); w.close()
