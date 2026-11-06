package decoder

import (
	"log/slog"

	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// SetLogger sends the library's operational messages to l: what a load decided, why a path declined, what the KV pool did, from this package
// and the GPU backends. Each is a record at a level (debug for routine bookkeeping, info for a decision, warn for something working worse
// than asked), and nothing reaches stderr. With no logger, or after SetLogger(nil), each is a line of plain text on stderr. It takes effect
// at once, for the whole process, and is safe to call at any time.
//
// Experimental (docs/api-tiers.md). It does not cover a server's own output when embedding server.NewHandler, which is goinfer-serve's banner
// and request log, or pull's download progress.
func SetLogger(l *slog.Logger) { logs.Set(l) }
