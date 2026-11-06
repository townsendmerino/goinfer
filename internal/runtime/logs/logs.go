// Package logs is where library code writes its operational messages: what a load decided, why a path declined, what the KV pool did. A
// binary's own output, its usage text and its answers on stdout, is the binary's, and stays out of here.
//
// Contract. With no logger set, a message is its text and a newline on stderr, whatever its level. That text is an interface, not decoration:
// scripts/bench_peer.py reads a server's fit and decode-path lines from its log, and a test that swaps os.Stderr sees it. Set, which
// decoder.SetLogger calls, hands everything to a *slog.Logger instead: the same text as the record's message, at the level the call names,
// attributed to the call's source line, and nothing on stderr. The Text functions take a string that already holds its own newlines, written
// as it is by default and trimmed for a logger.
//
// Invariants, each with the test that holds it:
//
//	the default is the text, byte for byte               TestDefault_isTheTextOnStderr
//	a logger gets the level, the text and the caller     TestLogger_getsLevelTextAndCaller
//	a level the logger drops costs no formatting         TestLogger_droppedLevelIsNotFormatted
//	nil restores the default                             TestSet_nilRestoresTheDefault
//	library code writes nowhere else                     TestLibraryOutput_goesThroughLogs
//	no block both reports an error and returns it        TestLibraryOutput_goesThroughLogs
package logs

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"time"
)

var logger atomic.Pointer[slog.Logger]

// Set sends every message to l; nil restores stderr. Safe to call while messages are being written.
func Set(l *slog.Logger) { logger.Store(l) }

// Debugf is routine bookkeeping an operator rarely wants: per-request KV moves.
func Debugf(format string, args ...any) { logf(slog.LevelDebug, format, args...) }

// Infof is what a load or a path decided, and a diagnostic an operator turned on (a timing or debug variable).
func Infof(format string, args ...any) { logf(slog.LevelInfo, format, args...) }

// Warnf is something working worse than asked: a decline, a fallback, a setting ignored.
func Warnf(format string, args ...any) { logf(slog.LevelWarn, format, args...) }

// DebugText, InfoText and WarnText are the levels for a string that already holds its own newlines.
func DebugText(text string) { writeText(slog.LevelDebug, text) }

// InfoText is Infof for a string that already holds its own newlines.
func InfoText(text string) { writeText(slog.LevelInfo, text) }

// WarnText is Warnf for a string that already holds its own newlines.
func WarnText(text string) { writeText(slog.LevelWarn, text) }

// logf loads the logger once, so a Set while it runs cannot hand a record to a logger whose level was not checked.
func logf(level slog.Level, format string, args ...any) {
	l := logger.Load()
	if l == nil {
		fmt.Fprintln(os.Stderr, fmt.Sprintf(format, args...))
		return
	}
	if l.Enabled(context.Background(), level) {
		record(l, level, fmt.Sprintf(format, args...))
	}
}

func writeText(level slog.Level, text string) {
	l := logger.Load()
	if l == nil {
		fmt.Fprint(os.Stderr, text)
		return
	}
	if msg := strings.TrimRight(text, "\n"); msg != "" && l.Enabled(context.Background(), level) {
		record(l, level, msg)
	}
}

// record must be called two frames below the level function: the source it reports is that function's caller.
func record(l *slog.Logger, level slog.Level, msg string) {
	var pcs [1]uintptr
	runtime.Callers(4, pcs[:]) // Callers, record, logf or writeText, the level function
	_ = l.Handler().Handle(context.Background(), slog.NewRecord(time.Now(), level, msg, pcs[0]))
}
