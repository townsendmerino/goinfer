package serveapp

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"time"

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
// The optional tiered-KV demotion (see enableTiering) runs under that same lock,
// so the background demoter and the request path never race.
type sessionLRU struct {
	model   *decoder.Model
	size    int
	capHint int
	fp      string             // model fingerprint stamped into / checked against snapshots
	adapter string             // compute-time LoRA adapter name bound to every session here (#7); "" = base
	order   []*decoder.Session // most-recently-used first; resident in RAM; len ≤ size

	// Tiered KV (idea #8): demote idle warm sessions RAM → NVMe. Zero value =
	// disabled, in which case behavior is exactly the original RAM-only LRU.
	// Enabled via enableTiering; demoteAfter > 0 is the on switch.
	demoteAfter time.Duration                  // idle threshold; a session untouched this long is demoted
	demotedMax  int                            // cap on the on-disk cold tier (older cold sessions are evicted)
	dir         string                         // where cold blobs are written (the model's -session-dir subdir)
	cold        []*coldSession                 // demoted (on-disk) sessions, most-recently-demoted first
	touched     map[*decoder.Session]time.Time // last-acquire time per resident session
	seq         int                            // monotonic counter for unique cold-blob filenames
	now         func() time.Time               // clock seam (tests override); defaults to time.Now
}

// coldSession is a demoted session: its KV lives in a .giw-kv blob on disk and its
// RAM is freed. Only its token sequence (so the LRU can still prefix-match it) and
// the blob path stay resident — a few hundred bytes versus the full cache.
type coldSession struct {
	tokens []int
	path   string
}

// KV snapshots and cold blobs are the CONVERSATION, not a cache of public data: a .giw-kv blob
// replays what the user said and what the model answered. They were written 0o644 inside a 0o755
// directory, so every local account could read them (audit-2026-09-02 N-21). Owner-only, both
// levels — the directory matters as much as the files, since a readable directory lists the
// session ids.
const (
	sessionDirPerm  = 0o700
	sessionFilePerm = 0o600
)

func newSessionLRU(m *decoder.Model, size, capHint int, fp string) *sessionLRU {
	if size < 0 {
		size = 0
	}
	return &sessionLRU{model: m, size: size, capHint: capHint, fp: fp, now: time.Now}
}

// newSession creates an empty session bound to this LRU's compute-time adapter
// (#7) — so every session it hands out projects through the right fine-tune. Base
// LRUs (adapter == "") get a plain session. UseAdapter can't fail here: the
// adapter was registered on the shared model at load before any LRU referenced it.
func (l *sessionLRU) newSession() *decoder.Session {
	s := l.model.NewSession(l.capHint)
	l.bindAdapter(s)
	return s
}

// bindAdapter activates this LRU's adapter on s (no-op for a base LRU). Applied to
// freshly created and fault-back-restored sessions alike, since a snapshot carries
// KV but not the active adapter.
func (l *sessionLRU) bindAdapter(s *decoder.Session) {
	if l.adapter != "" {
		_ = s.UseAdapter(l.adapter)
	}
}

// enableTiering turns on tiered KV: resident sessions idle longer than after are
// demoted to .giw-kv blobs under dir (freeing RAM) and faulted back on the next
// matching request; capacity evictions also tier the coldest session to disk
// instead of discarding it. dir is the model's -session-dir subdir; max bounds the
// on-disk tier (oldest cold sessions beyond it are deleted). Stale cold blobs from
// a previous run are removed — the cold tier is in-process (the resident tier is
// what save/load persists across restarts). No-op when after ≤ 0 or size == 0.
func (l *sessionLRU) enableTiering(dir string, after time.Duration, max int) {
	if after <= 0 || l.size == 0 {
		return
	}
	if max < 1 {
		max = 1
	}
	l.dir, l.demoteAfter, l.demotedMax = dir, after, max
	if l.touched == nil {
		l.touched = make(map[*decoder.Session]time.Time, l.size)
	}
	// Drop any orphaned cold blobs from a prior process (not in our index).
	old, _ := filepath.Glob(filepath.Join(dir, "cold-*"+sessionSnapExt))
	for _, p := range old {
		_ = os.Remove(p)
	}
}

// tiering reports whether tiered KV is active.
func (l *sessionLRU) tiering() bool { return l.demoteAfter > 0 }

// mark records that s was just acquired (for the idle-demotion clock). No-op when
// tiering is off.
func (l *sessionLRU) mark(s *decoder.Session) {
	if l.touched != nil {
		l.touched[s] = l.now()
	}
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
		return l.newSession()
	}
	lists := make([][]int, len(l.order))
	for i, s := range l.order {
		lists[i] = s.Tokens()
	}
	if i := bestExtend(lists, prompt); i >= 0 {
		moveToFront(l.order, i)
		l.mark(l.order[0])
		return l.order[0]
	}
	// No resident match. With tiered KV, the continuation may belong to a session
	// we demoted to disk — fault it back rather than re-prefilling from cold.
	if l.tiering() {
		if s := l.faultBack(prompt); s != nil {
			return s
		}
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
// the evicted-and-reset coldest session (reusing its backing arrays). With tiered
// KV the coldest is first demoted to disk (so its KV survives, faultable on a
// later continuation) and its now-empty backing arrays reused for the new slot.
func (l *sessionLRU) fresh() *decoder.Session {
	if len(l.order) < l.size {
		s := l.newSession()
		l.order = append([]*decoder.Session{s}, l.order...)
		l.mark(s)
		return s
	}
	if l.tiering() {
		if s, ok := l.tierOut(len(l.order) - 1); ok {
			s.Reset() // reuse the demoted session's backing arrays for the fresh slot
			l.order = append([]*decoder.Session{s}, l.order...)
			l.mark(s)
			return s
		}
		// coldest couldn't be tiered (non-persistable cache / IO error): fall
		// through to the original in-place reset eviction below.
	}
	s := l.order[len(l.order)-1] // coldest
	s.Reset()
	moveToFront(l.order, len(l.order)-1)
	l.mark(s)
	return s
}

// --- tiered KV: demote idle / overflow sessions to disk, fault back on demand ---

// demoteIdle moves every resident session untouched for longer than demoteAfter
// into the on-disk cold tier, freeing its RAM. Returns the number demoted. The
// server calls this from a background ticker under the model lock; it is a no-op
// when tiering is off. Sessions whose cache can't be serialized (a recurrent
// hybrid cache) are left resident.
func (l *sessionLRU) demoteIdle() int {
	if !l.tiering() {
		return 0
	}
	cutoff := l.now().Add(-l.demoteAfter)
	n := 0
	// Walk coldest→newest and remove in place; tierOut deletes index i, leaving
	// the lower indices we haven't visited yet unchanged.
	for i, v := range slices.Backward(l.order) {
		t, ok := l.touched[v]
		if ok && t.After(cutoff) {
			continue // still warm
		}
		if _, done := l.tierOut(i); done {
			n++ // freed session dropped (RAM reclaimed by GC), not reused
		}
	}
	return n
}

// tierOut snapshots order[i] to a fresh cold blob, removes it from the resident
// order, and records it in the cold tier. It returns the removed *decoder.Session
// (whose backing arrays the caller may Reset+reuse) and true on success. It
// returns (nil, false) without modifying order when the session is empty, its
// cache is non-persistable, or the write fails — the caller then keeps/evicts it
// the ordinary way.
func (l *sessionLRU) tierOut(i int) (*decoder.Session, bool) {
	s := l.order[i]
	if len(s.Tokens()) == 0 {
		return nil, false // nothing worth preserving
	}
	blob := s.Snapshot(l.fp)
	if blob == nil {
		return nil, false // recurrent/hybrid cache: not persistable (Snapshot refuses)
	}
	if err := os.MkdirAll(l.dir, sessionDirPerm); err != nil {
		fmt.Fprintf(os.Stderr, "tiered-kv: mkdir %s: %v\n", l.dir, err)
		return nil, false
	}
	l.seq++
	path := filepath.Join(l.dir, fmt.Sprintf("cold-%06d%s", l.seq, sessionSnapExt))
	if err := os.WriteFile(path, blob, sessionFilePerm); err != nil {
		fmt.Fprintf(os.Stderr, "tiered-kv: write %s: %v\n", path, err)
		return nil, false
	}
	l.cold = append([]*coldSession{{tokens: append([]int(nil), s.Tokens()...), path: path}}, l.cold...)
	l.order = append(l.order[:i], l.order[i+1:]...)
	delete(l.touched, s)
	l.evictColdOverflow()
	return s, true
}

// faultBack restores a demoted session whose tokens are a prefix of prompt, making
// room in the resident tier (tiering the coldest, or evicting it if that fails),
// and returns it marked most-recently-used. nil if no cold session matches or its
// blob is unreadable/stale (caller falls back to a cold prefill).
func (l *sessionLRU) faultBack(prompt []int) *decoder.Session {
	lists := make([][]int, len(l.cold))
	for i, c := range l.cold {
		lists[i] = c.tokens
	}
	i := bestExtend(lists, prompt)
	if i < 0 {
		return nil
	}
	c := l.cold[i]
	s, err := l.readCold(c)
	l.dropCold(i) // remove from index + delete blob whether or not it loaded
	if err != nil {
		fmt.Fprintf(os.Stderr, "tiered-kv: fault-back %s: %v\n", filepath.Base(c.path), err)
		return nil
	}
	l.makeRoom()
	l.order = append([]*decoder.Session{s}, l.order...)
	l.mark(s)
	return s
}

// readCold deserializes a cold session's blob back into a live *decoder.Session.
func (l *sessionLRU) readCold(c *coldSession) (*decoder.Session, error) {
	data, err := os.ReadFile(c.path)
	if err != nil {
		return nil, err
	}
	s, err := l.model.LoadSession(data, l.fp)
	if err != nil {
		return nil, err
	}
	l.bindAdapter(s) // the snapshot carries KV but not the active adapter (#7)
	return s, nil
}

// makeRoom ensures the resident tier has space for one more session, tiering the
// coldest to disk (preferred) or, if that fails, evicting it outright.
func (l *sessionLRU) makeRoom() {
	if len(l.order) < l.size {
		return
	}
	if _, ok := l.tierOut(len(l.order) - 1); ok {
		return
	}
	i := len(l.order) - 1 // couldn't tier: drop the coldest to free the slot
	delete(l.touched, l.order[i])
	l.order = l.order[:i]
}

// dropCold removes cold[i] from the index and deletes its blob.
func (l *sessionLRU) dropCold(i int) {
	_ = os.Remove(l.cold[i].path)
	l.cold = append(l.cold[:i], l.cold[i+1:]...)
}

// evictColdOverflow trims the cold tier to demotedMax, deleting the oldest blobs.
func (l *sessionLRU) evictColdOverflow() {
	for len(l.cold) > l.demotedMax {
		l.dropCold(len(l.cold) - 1)
	}
}

// removeColdFiles deletes every cold blob this LRU has on disk (in-process tier
// teardown — the resident tier is what save persists across restarts).
func (l *sessionLRU) removeColdFiles() {
	for i := range l.cold {
		_ = os.Remove(l.cold[i].path)
	}
	l.cold = nil
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
	if err := os.MkdirAll(dir, sessionDirPerm); err != nil {
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
			// Snapshot refuses RECURRENT state (Mamba-2 / DeltaNet / LFM2 conv / MLA latent),
			// which cannot be restored from a KV blob. It used to refuse sliding-window rings
			// too, and this line still said so — rings have been persistable since
			// kvSnapVersion 2 (N-34).
			continue
		}
		p := filepath.Join(dir, fmt.Sprintf("session-%02d%s", i, sessionSnapExt))
		if err := os.WriteFile(p, blob, sessionFilePerm); err != nil {
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
		l.bindAdapter(s) // restored KV is base-projected bytes; re-activate the adapter (#7)
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
