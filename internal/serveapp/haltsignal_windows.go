//go:build windows

package serveapp

// startHaltSignalLoop is Windows' no-op twin of haltsignal_unix.go: SIGUSR1 and SIGUSR2 are undefined identifiers
// there, and no signal substitute exists for a supervisor-triggered halt or resume, so Windows uses -halt-file and
// the admin socket instead (docs/tasks/task-halt-2026-09.md).
func startHaltSignalLoop(srv *server) {}
