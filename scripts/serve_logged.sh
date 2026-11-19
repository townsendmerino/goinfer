#!/bin/bash
# serve_logged.sh: a goinfer server that keeps its log, whoever starts it. Reached through a symlink named <binary>.logged, it runs
# <binary> as this same process (exec: same pid, same arguments), with stdout and stderr each still on its own stream (a caller such as
# tack reads the banner from stderr), and appends both to a log: $SERVE_LOG when set, else $NIGHT_RUNDIR/$NIGHT_JOB.serve.log inside a
# night.py job, else <binary>.serve.log beside the binary.
#
#   ln -s <repo>/scripts/serve_logged.sh ~/tack-runs/bin/goinfer-serve-cuda-<rev>.logged
#   then point the manifest's goinfer_serve at ~/tack-runs/bin/goinfer-serve-cuda-<rev>.logged
set -u
bin="${0%.logged}"
if [ "$bin" = "$0" ] || [ ! -x "$bin" ]; then
  echo "serve_logged.sh: reach it as <binary>.logged beside an executable <binary> (got $0)" >&2
  exit 2
fi
if [ -n "${SERVE_LOG:-}" ]; then
  log=$SERVE_LOG
elif [ -n "${NIGHT_RUNDIR:-}" ] && [ -n "${NIGHT_JOB:-}" ]; then
  log=$NIGHT_RUNDIR/$NIGHT_JOB.serve.log
else
  log=$bin.serve.log
fi
echo "=== $(date '+%F %T %Z') $bin $*" >> "$log"
exec 1> >(tee -a "$log") 2> >(tee -a "$log" >&2)
exec "$bin" "$@"
