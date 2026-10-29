#!/bin/bash
# Runs under the exclusive lock: refuse (75) a heavy-load start, else run the test.
l1=$(sysctl -n vm.loadavg | awk '{print $2}')
echo "== inside the lock $(date '+%T'), load $(sysctl -n vm.loadavg)"
pmset -g batt | head -1
if ! awk "BEGIN{exit !($l1 < 4.0)}"; then echo "== heavy load at the start ($l1): not run"; exit 75; fi
exec $HOME/.claude/jobs/756f841f/tmp/bin/metal.test -test.run '^TestCost_metalOffCost$' -test.v -test.count=1 -test.timeout 10m
