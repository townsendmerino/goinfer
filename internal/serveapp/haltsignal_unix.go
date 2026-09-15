//go:build !windows

package serveapp

import (
	"os"
	"os/signal"
	"syscall"
)

// startHaltSignalLoop wires SIGUSR1 (halt) / SIGUSR2 (resume) for the life of the process — a
// supervisor can pull this switch without opening a socket or an HTTP client (K2,
// docs/tasks/task-halt-2026-09.md). Every unix target goinfer builds for has these signals;
// Windows has neither, and there is no signal-based substitute there (see
// haltsignal_windows.go's no-op twin) — a Windows supervisor uses -halt-file or the admin
// socket instead. The switch's case values are the reason this whole loop, not just the
// signal.Notify call, has to live in a platform file: syscall.SIGUSR1/SIGUSR2 are undefined
// identifiers on Windows, not merely signals Windows never raises.
func startHaltSignalLoop(srv *server) {
	haltSig := make(chan os.Signal, 1)
	signal.Notify(haltSig, syscall.SIGUSR1, syscall.SIGUSR2)
	go func() {
		for s := range haltSig {
			switch s {
			case syscall.SIGUSR1:
				srv.halt("SIGUSR1", "SIGUSR1")
			case syscall.SIGUSR2:
				srv.resume("SIGUSR2")
			}
		}
	}()
}
