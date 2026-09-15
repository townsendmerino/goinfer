//go:build windows

package serveapp

// startHaltSignalLoop is windows' no-op twin of haltsignal_unix.go: SIGUSR1/SIGUSR2 do not
// exist on Windows (the identifiers themselves are undefined there, not merely signals Windows
// never raises), and there is no signal-based substitute for a supervisor-triggered halt/resume
// on this platform — -halt-file and the admin socket are the mechanisms Windows has instead
// (K2, docs/tasks/task-halt-2026-09.md).
func startHaltSignalLoop(srv *server) {}
