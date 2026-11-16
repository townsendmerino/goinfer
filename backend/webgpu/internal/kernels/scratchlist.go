//go:build gpu

package kernels

import "sort"

// ScratchEntry is one named group of a decode runner's build-time scratch leases: the role that created them (the op builder or build step that
// asked for them), how many buffers and their bytes. The runner builds its scratch from op builders that each name themselves, so the list is
// the whole of what the build leases on the ledger's scratch line (MM5b: the plan's figure for it is held to this list's sum).
type ScratchEntry struct {
	Role  string
	Count int
	Bytes int64
}

// scratchBook accumulates ScratchEntry by role while a runner is built.
type scratchBook struct {
	byRole map[string]*ScratchEntry
}

func (b *scratchBook) add(role string, bytes int64) {
	if b.byRole == nil {
		b.byRole = map[string]*ScratchEntry{}
	}
	e := b.byRole[role]
	if e == nil {
		e = &ScratchEntry{Role: role}
		b.byRole[role] = e
	}
	e.Count++
	e.Bytes += bytes
}

// list is the entries in role order.
func (b *scratchBook) list() []ScratchEntry {
	out := make([]ScratchEntry, 0, len(b.byRole))
	for _, e := range b.byRole {
		out = append(out, *e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Role < out[j].Role })
	return out
}

// ScratchList is the runner's build-time scratch leases by role. A lease no role names is listed under the empty role.
func (r *DecodeRunner) ScratchList() []ScratchEntry { return r.scratch.list() }

// ScratchBytes is the sum of ScratchList.
func (r *DecodeRunner) ScratchBytes() int64 {
	var n int64
	for _, e := range r.scratch.byRole {
		n += e.Bytes
	}
	return n
}
