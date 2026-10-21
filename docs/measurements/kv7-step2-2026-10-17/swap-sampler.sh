#!/usr/bin/env bash
# swap-sampler.sh OUT [SECONDS=10] — every interval: system swap in use, the pinned server's VmSwap/VmRSS/VmHWM, and the five processes holding the most VmSwap.
OUT=$1; IV=${2:-10}
echo "t_s,swap_used_mb,srv_vmswap_mb,srv_vmrss_mb,srv_vmhwm_mb,top5_vmswap_mb(pid:comm)" > "$OUT"
t0=$(date +%s)
while :; do
  pid=$(pgrep -f '^/home/francis/goinfer-bench/kv7-step2-2026-10/bin/serve-cuda-kv7s2' | head -1)
  used=$(awk '/^SwapTotal/{t=$2}/^SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  s=""; [ -n "$pid" ] && s=$(awk '/VmSwap/{a=int($2/1024)}/VmRSS/{b=int($2/1024)}/VmHWM/{c=int($2/1024)}END{print a","b","c}' /proc/$pid/status 2>/dev/null)
  top=$(for p in /proc/[0-9]*; do v=$(awk '/VmSwap/{print $2}' $p/status 2>/dev/null); [ -n "$v" ] && [ "$v" -gt 0 ] && echo "$((v/1024)) ${p#/proc/}:$(cat $p/comm 2>/dev/null)"; done | sort -rn | head -5 | tr '\n' ' ')
  echo "$(( $(date +%s)-t0 )),$used,${s:-,,},$top" >> "$OUT"
  sleep $IV
done
