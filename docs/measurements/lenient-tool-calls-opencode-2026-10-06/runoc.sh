#!/bin/bash
# usage: runoc.sh <label>
cd /tmp/claude-1000/coder/proj && cp ../calc.go.orig calc.go
export XDG_CONFIG_HOME=/tmp/claude-1000/coder/xdg/config XDG_DATA_HOME=/tmp/claude-1000/coder/xdg/data XDG_CACHE_HOME=/tmp/claude-1000/coder/xdg/cache
timeout 240 /tmp/claude-1000/coder/oc/node_modules/.bin/opencode run "${PROMPT:-fix the bug in calc.go}" </dev/null > ../oc-$1.log 2>&1
echo "rc=$?" >> ../oc-$1.log
echo "--- calc.go after:" >> ../oc-$1.log; cat calc.go >> ../oc-$1.log
