# Shared by the Metal KV8a runners (README.md here). Sourced, not run.
K=$HOME/goinfer-bench/kv8a-metal; H=$HOME/goinfer-bench/kv0h-2026-10
KV8A=$K/bin/serve-metal-9488f592; BEFORE=$K/bin/serve-metal-a8b1c806
MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.int4.metal.v17.giw
MODEL_ARG="Qwen3.5-9B-Q4_K_M=$MODEL" # served under the name the recorded bodies ask for (a bare .giw path is served as its file name)
NEED_GB=6.7 # the void rule's fit need for the 9B: weights 5.5 GB + KV 1.2 GB at -ctx 20480, from its fit banner (2026-10-21)

free_gb() { # the fit guard's reading (decoder.HostRAMAvailableBytes): free + inactive + speculative + purgeable pages
	vm_stat | python3 -c 'import re,sys; o=sys.stdin.read(); ps=int(re.search(r"page size of (\d+) bytes",o).group(1)); print("%.2f" % (sum(int(re.search(re.escape(k)+r"\s+(\d+)\.",o).group(1)) for k in ("Pages free:","Pages inactive:","Pages speculative:","Pages purgeable:"))*ps/2**30))'
}
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }

# start_clean WAIT_S: the heavy-load void rule at the start (load average <= 4.0, half this Mac's cores; free memory >= NEED_GB + 2),
# retried every 20 s for WAIT_S seconds. Prints each reading; returns 0 on a clean start, 1 when it never got one.
start_clean() {
	local waited=0 la fr
	while :; do
		la=$(load1); fr=$(free_gb)
		echo "start check: load $la (void above 4.0), free memory $fr GB (void under $(python3 -c "print($NEED_GB+2)") GB), swap $(sysctl -n vm.swapusage | awk '{print $6}')"
		python3 -c "import sys; sys.exit(0 if $la <= 4.0 and $fr >= $NEED_GB + 2 else 1)" && return 0
		[ "$waited" -ge "$1" ] && return 1
		sleep 20; waited=$((waited + 20))
	done
}

# purge_then_check WAIT_S: drops the file cache with `sudo -n /usr/sbin/purge` (Francis's /etc/sudoers.d/purge, 2026-10-21: that one
# command, passwordless), recording the reading before and after, then start_clean. A model's own cached file pages are filed as active
# after it exits, so the reading counts them as used until something purges them (README.md here, "The day checks"). The rule and the
# 8.7 GB need are unchanged. Returns 1 when the purge fails (that start is void) or the start is not clean.
purge_then_check() {
	echo "before purge: free memory $(free_gb) GB, load $(load1)"
	if ! sudo -n /usr/sbin/purge; then
		echo "VOID start: sudo -n /usr/sbin/purge failed"
		return 1
	fi
	echo "after purge: free memory $(free_gb) GB, load $(load1)"
	start_clean "$1"
}

# fit_check CTX: one load of the KV8a build at CTX; prints its decode path, fit, KV plan and checkpoint lines. Returns 3 when the
# load is not Metal-resident (the guard refused it).
fit_check() {
	local log=$OUT/fitcheck-$1.log pid
	"$KV8A" -model "m=$MODEL" -backend metal -addr 127.0.0.1:18302 -ctx "$1" --kv-sessions 1 > "$log" 2>&1 &
	pid=$!
	for _ in $(seq 1 300); do
		grep -q 'goinfer serving on' "$log" && break
		kill -0 $pid 2>/dev/null || break
		sleep 1
	done
	kill $pid 2>/dev/null; wait $pid 2>/dev/null
	echo "-- fit check at -ctx $1:"; grep -E 'decode path|fit:|KV plan|recurrent checkpoint|declined|refus' "$log" | cut -c1-240
	grep -q 'decode path: metal-resident' "$log" || return 3
}
