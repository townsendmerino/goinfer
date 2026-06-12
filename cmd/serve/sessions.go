package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/townsendmerino/goinfer/decoder"
)

// sessionLRU keeps up to size prefilled KV sessions and hands each request the
// one that already holds its prompt as a prefix — so a continuing chat (or an
// agent loop with a fixed system prompt + tool specs) prefills only the new
// suffix instead of the whole history. It is the cmd/serve layer over
// decoder.Session; the decoder does the exact prefix reuse, the LRU just decides
// which session to reuse and which to evict.
//
// Not goroutine-safe. The server holds s.mu across every generation, which
// serializes all access — the same lock that already serializes the shared model.
type sessionLRU struct {
	model   *decoder.Model
	size    int
	capHint int
	fp      string             // model fingerprint stamped into / checked against snapshots
	order   []*decoder.Session // most-recently-used first; len ≤ size
}

func newSessionLRU(m *decoder.Model, size, capHint int, fp string) *sessionLRU {
	if size < 0 {
		size = 0
	}
	return &sessionLRU{model: m, size: size, capHint: capHint, fp: fp}
}

// acquire returns the session to generate prompt against and marks it most-
// recently-used. It reuses a session only when prompt extends that session's
// entire stored sequence (its tokens are a prefix of prompt) — the chat/agent
// continuation case, where reuse is a pure win with no KV thrown away. A prompt
// that merely shares a system-prompt preamble with some other conversation's
// session does NOT hijack it (that session's later turns aren't a prefix of this
// prompt), so distinct conversations keep their own slots. Anything else gets a
// fresh slot, evicting the coldest session when full.
//
// With size 0 reuse is disabled: every call gets a throwaway session (the old
// re-prefill-everything behavior).
func (l *sessionLRU) acquire(prompt []int) *decoder.Session {
	if l.size == 0 {
		return l.model.NewSession(l.capHint)
	}
	lists := make([][]int, len(l.order))
	for i, s := range l.order {
		lists[i] = s.Tokens()
	}
	if i := bestExtend(lists, prompt); i >= 0 {
		moveToFront(l.order, i)
		return l.order[0]
	}
	return l.fresh()
}

// bestExtend returns the index of the session whose entire token sequence is a
// prefix of prompt — the continuation case where reuse is a pure win with no KV
// discarded — preferring the longest (most reused). -1 if none qualifies. A
// session whose tokens are NOT fully contained in prompt (a different
// conversation that merely shares a system-prompt preamble) is never chosen, so
// distinct conversations don't evict each other's tails.
func bestExtend(sessions [][]int, prompt []int) int {
	best, bestLen := -1, 0
	for i, toks := range sessions {
		n := len(toks)
		if n > bestLen && n == commonPrefix(toks, prompt) {
			best, bestLen = i, n
		}
	}
	return best
}

// fresh returns an empty session at the front: a new one if there's room, else
// the evicted-and-reset coldest session (reusing its backing arrays).
func (l *sessionLRU) fresh() *decoder.Session {
	if len(l.order) < l.size {
		s := l.model.NewSession(l.capHint)
		l.order = append([]*decoder.Session{s}, l.order...)
		return s
	}
	s := l.order[len(l.order)-1] // coldest
	s.Reset()
	moveToFront(l.order, len(l.order)-1)
	return s
}

// moveToFront promotes s[i] to index 0, shifting the rest right (LRU recency).
func moveToFront[T any](s []T, i int) {
	if i <= 0 {
		return
	}
	v := s[i]
	copy(s[1:i+1], s[:i])
	s[0] = v
}

// commonPrefix returns the length of the longest common leading run of a and b.
func commonPrefix(a, b []int) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// --- disk persistence (-session-dir) ---

const sessionSnapExt = ".giw-kv"

// save writes each live session to dir as a CRC-guarded .giw-kv snapshot,
// replacing any prior snapshots there. Each is stamped with the model
// fingerprint so a later load against a different model rejects it. Best-effort:
// a failed write is logged, not fatal — a snapshot is a cache, never a source of
// truth.
func (l *sessionLRU) save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// Drop stale snapshots first so an eviction since last save doesn't leave an
	// orphan that reloads as a phantom session.
	old, _ := filepath.Glob(filepath.Join(dir, "session-*"+sessionSnapExt))
	for _, p := range old {
		_ = os.Remove(p)
	}
	saved := 0
	for i, s := range l.order {
		if len(s.Tokens()) == 0 {
			continue
		}
		blob := s.Snapshot(l.fp)
		if blob == nil {
			continue // sliding-window (ring) cache: not yet persistable (Inc 3)
		}
		p := filepath.Join(dir, fmt.Sprintf("session-%02d%s", i, sessionSnapExt))
		if err := os.WriteFile(p, blob, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "session snapshot %s: %v\n", p, err)
			continue
		}
		saved++
	}
	fmt.Fprintf(os.Stderr, "saved %d KV session(s) to %s\n", saved, dir)
	return nil
}

// load restores sessions from dir's .giw-kv snapshots (most positions first, up
// to size). Snapshots from a different model/geometry are skipped with a log,
// not an error. Returns the number loaded.
func (l *sessionLRU) load(dir string) int {
	paths, _ := filepath.Glob(filepath.Join(dir, "session-*"+sessionSnapExt))
	sort.Strings(paths) // session-00 first
	loaded := 0
	for _, p := range paths {
		if len(l.order) >= l.size {
			break
		}
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		s, err := l.model.LoadSession(data, l.fp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", filepath.Base(p), err)
			continue
		}
		l.order = append(l.order, s)
		loaded++
	}
	if loaded > 0 {
		fmt.Fprintf(os.Stderr, "restored %d KV session(s) from %s\n", loaded, dir)
	}
	return loaded
}

// sessionDirOK rejects an obviously-unsafe -session-dir (a path that exists but
// isn't a directory).
func sessionDirOK(dir string) error {
	if dir == "" {
		return nil
	}
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		return fmt.Errorf("-session-dir %q is not a directory", dir)
	}
	return nil
}
