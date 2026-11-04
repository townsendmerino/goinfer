//go:build !windows

package serveapp

import (
	"os"
	"os/signal"
	"syscall"
)

// startHaltSignalLoop wires SIGUSR1 (halt) and SIGUSR2 (resume) for the life of the process, so a supervisor can pull
// the switch without a socket or an HTTP client (docs/tasks/task-halt-2026-09.md). Windows has neither signal and no
// substitute (see haltsignal_windows.go's no-op twin): a Windows supervisor uses -halt-file or the admin socket. The
// whole loop, not just the signal.Notify call, lives in a platform file because the switch's case values,
// syscall.SIGUSR1 and SIGUSR2, are undefined identifiers on Windows.
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
