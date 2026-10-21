#!/usr/bin/env bash
# install-night-agent.sh — runs `night.py serve` (the always-on runner, B38) under the box's service manager: a launchd agent on macOS,
# a systemd user service on Linux. It runs from the pinned runner checkout ($NIGHT_RUNNER_DIR, default ~/goinfer-runner), which
# `night.py update-runner` creates and moves; nobody edits it.
#
#   scripts/install-night-agent.sh print      write the unit to stdout and lint it; installs nothing
#   scripts/install-night-agent.sh install    write, lint, load and start it
#   scripts/install-night-agent.sh uninstall  stop and remove it (queued jobs stay queued)
#   scripts/install-night-agent.sh status     what the service manager says, then `night.py status`
#
# Jobs inherit the PATH this script is run with; the hooks file's path_prepend still goes in front (night.py's docstring).
set -euo pipefail

RUNNER_DIR=${NIGHT_RUNNER_DIR:-$HOME/goinfer-runner}
NIGHT_DIR=${NIGHT_DIR:-$HOME/goinfer-logs/night}
PY=$(command -v python3)
NIGHT=$RUNNER_DIR/scripts/night.py
LABEL=com.goinfer.night
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
UNIT=goinfer-night.service
UNIT_FILE=$HOME/.config/systemd/user/$UNIT

xml() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }

plist() {
	cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>$LABEL</string>
	<key>ProgramArguments</key>
	<array><string>$(xml "$PY")</string><string>$(xml "$NIGHT")</string><string>serve</string></array>
	<key>WorkingDirectory</key><string>$(xml "$NIGHT_DIR")</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PATH</key><string>$(xml "$PATH")</string>
		<key>NIGHT_DIR</key><string>$(xml "$NIGHT_DIR")</string>
	</dict>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><true/>
	<key>ThrottleInterval</key><integer>30</integer>
	<key>ProcessType</key><string>Standard</string>
	<key>StandardOutPath</key><string>$(xml "$NIGHT_DIR/serve.log")</string>
	<key>StandardErrorPath</key><string>$(xml "$NIGHT_DIR/serve.log")</string>
</dict>
</plist>
EOF
}

unit() {
	cat <<EOF
[Unit]
Description=goinfer night.py serve (the always-on queue runner)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$NIGHT_DIR
Environment="PATH=$PATH" "NIGHT_DIR=$NIGHT_DIR"
ExecStart=$PY $NIGHT serve
Restart=always
RestartSec=30
# SIGTERM to serve alone: it kills the running job's process group itself and re-queues it, then exits.
KillMode=mixed
TimeoutStopSec=90
StandardOutput=append:$NIGHT_DIR/serve.log
StandardError=append:$NIGHT_DIR/serve.log

[Install]
WantedBy=default.target
EOF
}

need_runner() {
	[ -f "$NIGHT" ] || { echo "no runner checkout at $RUNNER_DIR: run 'python3 scripts/night.py update-runner' first" >&2; exit 1; }
}

lint() { # $1: the written file
	case $(uname) in
	Darwin) plutil -lint "$1" >/dev/null ;;
	*) if command -v systemd-analyze >/dev/null; then systemd-analyze --user verify "$1"; fi ;;
	esac
}

cmd=${1:-}
mkdir -p "$NIGHT_DIR"
case "$(uname):$cmd" in
Darwin:print)
	tmp=$(mktemp -t night-agent).plist
	plist >"$tmp"
	cat "$tmp"
	lint "$tmp" && echo "plutil: OK" >&2
	rm -f "$tmp" ;;
Linux:print)
	d=$(mktemp -d)
	unit >"$d/$UNIT"
	cat "$d/$UNIT"
	lint "$d/$UNIT" && echo "systemd-analyze verify: OK" >&2
	rm -rf "$d" ;;
Darwin:install)
	need_runner
	mkdir -p "$(dirname "$PLIST")"
	plist >"$PLIST.new"
	lint "$PLIST.new"
	mv "$PLIST.new" "$PLIST"
	launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
	launchctl bootstrap "gui/$(id -u)" "$PLIST"
	echo "installed $PLIST; log $NIGHT_DIR/serve.log" ;;
Linux:install)
	need_runner
	mkdir -p "$(dirname "$UNIT_FILE")"
	unit >"$UNIT_FILE.new"
	mv "$UNIT_FILE.new" "$UNIT_FILE"
	lint "$UNIT_FILE"
	systemctl --user daemon-reload
	systemctl --user enable --now "$UNIT"
	if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]; then
		echo "NOTE: lingering is off, so the service stops at logout. 'loginctl enable-linger $USER' keeps it up (needs the owner)." >&2
	fi
	echo "installed $UNIT_FILE; log $NIGHT_DIR/serve.log" ;;
Darwin:uninstall)
	launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
	rm -f "$PLIST"
	echo "removed $LABEL (queued jobs stay queued)" ;;
Linux:uninstall)
	systemctl --user disable --now "$UNIT" 2>/dev/null || true
	rm -f "$UNIT_FILE"
	systemctl --user daemon-reload
	echo "removed $UNIT (queued jobs stay queued)" ;;
Darwin:status)
	launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E '^\s*(state|pid|last exit code) ' || echo "$LABEL: not loaded"
	"$PY" "$NIGHT" status ;;
Linux:status)
	systemctl --user --no-pager status "$UNIT" | head -5 || true
	"$PY" "$NIGHT" status ;;
*)
	sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
	exit 2 ;;
esac
