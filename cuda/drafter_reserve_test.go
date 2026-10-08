//go:build cuda && goinfer_testhooks

package cuda

import "github.com/townsendmerino/goinfer/decoder"

// withDrafterReserve is o for a target a block drafter attaches to AFTER the load, priced the way `serve --drafter` prices it (internal/serveapp/main.go loads the
// drafter first and sets both fields): the plan then leaves the drafter's int8 weights and its per-position K/V out of the context it chooses. A test that loads the
// target plainly and bolts NewBlockSpec on afterwards gets a context sized to every byte beyond the 384 MiB margin and the drafter's allocation runs out of device
// memory (the 2026-10-07 night gate's three drafter failures, root-caused 2026-10-08: the loads logged "less 384 MB reserved", the margin alone). Production never
// does that, so the tests were the stale half.
func withDrafterReserve(o decoder.Options, dr *decoder.DFlashDrafter) decoder.Options {
	o.ExtraResidentBytes = decoder.DrafterResidentBytesEstimate(dr)
	o.ExtraResidentKVPerPosition = decoder.DrafterKVBytesPerPosition(dr)
	return o
}
