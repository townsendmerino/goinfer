//go:build windows

package serveapp

// startHaltSignalLoop is Windows' no-op twin of haltsignal_unix.go: SIGUSR1 and SIGUSR2 are undefined identifiers
// there, and no signal substitute exists for a supervisor-triggered halt or resume, so Windows uses -halt-file and
// the admin socket instead.
func startHaltSignalLoop(srv *server) {}
