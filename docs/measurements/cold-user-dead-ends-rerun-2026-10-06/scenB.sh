#!/bin/bash
# usage: scenB.sh <label> <serve-binary>   — scenario B on Qwen2.5-7B-Instruct: serve check + opencode runs (A: vague x3, B: explicit x8)
LABEL=$1; BIN=$2; D=/tmp/claude-1000/scen; OUT=$D/B-$LABEL; mkdir -p $OUT
M=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
echo "[$(date +%T)] $LABEL: $($BIN --version 2>&1 | head -1)"
(setsid nohup $BIN -addr 127.0.0.1:18084 -model $M -quant int4 -backend cuda -ctx 16384 > $OUT/serve.log 2>&1 < /dev/null & echo $! > $OUT/serve.pid)
for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:18084/v1/models && break; sleep 3; done
grep -i "decode path\|context:" $OUT/serve.log | head -2
echo "[$(date +%T)] serve check"; $BIN check http://127.0.0.1:18084 > $OUT/check.txt 2>&1; echo "check rc=$?"; sed 's/\x1b\[[0-9;]*m//g' $OUT/check.txt | grep -E "ok|FAIL|skip" | cut -c1-120
export XDG_CONFIG_HOME=$D/xdg/config XDG_DATA_HOME=$D/xdg/data XDG_CACHE_HOME=$D/xdg/cache
run() { # run <tag> <prompt>
  cd $D/projB && cp ../calc.go.orig calc.go
  timeout 300 /tmp/claude-1000/scen/oc/node_modules/.bin/opencode run "$2" </dev/null > $OUT/oc-$1.log 2>&1
  r=$(grep -E 'return a' calc.go | tr -d '\t '); echo "[$(date +%T)] $1: $r"
  cp calc.go $OUT/calc-$1.go
}
for i in 1 2 3; do run A$i "fix the bug in calc.go"; done
PB='Fix the bug in calc.go. First use the read tool to read calc.go, then use the edit tool to change it. Do not describe the change; make it with the edit tool.'
for i in 1 2 3 4 5 6 7 8; do run B$i "$PB"; done
kill "$(cat $OUT/serve.pid)"; sleep 3
echo "[$(date +%T)] $LABEL done"
