package decoder

// The cache-state grid (docs/tasks/task-option-path-admission-2026-10.md §4.0): every kind of per-sequence state a
// KVCache can hold, against every lifecycle path that has to know about it, with what that path does with it. It
// guards against a kind of state a lifecycle path never accounted for (Mamba-2 and DeltaNet state across a reset,
// LFM2's conv window, Bailing Hybrid's KDA state, the adapter that built a reused prefix), which an audit would
// otherwise find after it shipped, one path at a time (docs/measurements/audit-classes-2026-10-08.md).
//
// Two things make the grid fail closed rather than document intent:
//
//   - TestKVCache_everyFieldHasAState fails the commit that adds a KVCache field without saying
//     which kind of state it is (or that it is geometry, not state), and
//     TestCacheStateGrid_everyCellDeclared fails a kind with an undeclared lifecycle cell. A sixth
//     recurrent kind cannot land the way KDA did.
//   - The paths READ the grid: Snapshot refuses, and a partial TruncateTo reports inexact, for
//     every kind the grid marks so (holdsStateHandled), instead of each site hand-listing kinds.
//
// The behaviour behind each declaration is checked against real caches in cachestate_test.go, so
// a cell that says "refused" and is not fails there.

// cacheState is one kind of state a KVCache can hold.
type cacheState string

const (
	statePositional  cacheState = "positional KV"        // keys/vals per position on global (append-forever) layers
	stateRing        cacheState = "sliding-window ring"  // local layers' physical rings (rings, localAny)
	stateInt8KV      cacheState = "int8 KV + scales"     // keysQ/valsQ/keyScale/valScale under KVQuant i8
	stateMLALatent   cacheState = "MLA latent"           // DeepSeek/Kimi compressed-KV latent per position
	stateDeltaNet    cacheState = "DeltaNet state"       // Gated DeltaNet conv window + matrix state (recurrent)
	stateShortConv   cacheState = "short-conv window"    // LFM2's rolling conv window (recurrent)
	stateMamba2      cacheState = "Mamba-2 state"        // conv window + SSM state (recurrent)
	stateKDA         cacheState = "KDA state"            // Bailing Hybrid's three conv windows + matrix state (recurrent)
	stateImageBlocks cacheState = "image blocks"         // VL position ranges that attend bidirectionally
	stateMRoPE       cacheState = "m-RoPE positions"     // Qwen2.5-VL per-position (t,h,w) rotary positions + delta
	stateDeepstack   cacheState = "deepstack rows"       // Qwen3-VL visual rows added after early layers during the image prefill
	stateCapture     cacheState = "capture rows"         // per-forward hidden-state capture (EAGLE, sub-layer test hooks)
	stateTree        cacheState = "tree-verify mask"     // tree attention mask and row positions (tested plumbing; unset in production)
	stateAdapter     cacheState = "LoRA adapter binding" // the compute-time adapter this stream projects through
)

// cacheStates is every kind, in the grid's display order. TestCacheStateGrid_everyCellDeclared
// fails a kind that is declared in the grid but missing here, or the reverse.
var cacheStates = []cacheState{
	statePositional, stateRing, stateInt8KV, stateMLALatent,
	stateDeltaNet, stateShortConv, stateMamba2, stateKDA,
	stateImageBlocks, stateMRoPE, stateDeepstack,
	stateCapture, stateTree, stateAdapter,
}

// lifecyclePath is a place a sequence's state must survive, be cleared, or be refused.
type lifecyclePath string

const (
	lcRewind       lifecyclePath = "partial rewind"       // KVCache.TruncateTo(p), 0 < p < Pos()
	lcReset        lifecyclePath = "full reset"           // KVCache.TruncateTo(0): Session.Reset, sessionLRU eviction
	lcSnapshot     lifecyclePath = "snapshot"             // Session.Snapshot / Model.LoadSession
	lcReuse        lifecyclePath = "session prefix reuse" // Session.rewindForReuse
	lcSpecRollback lifecyclePath = "speculative rollback" // Model.specRollbackSafe and the rollback TruncateTo sites
)

// lifecyclePaths is every path, in display order.
var lifecyclePaths = []lifecyclePath{lcRewind, lcReset, lcSnapshot, lcReuse, lcSpecRollback}

// stateHandling is what a lifecycle path does with a kind of state.
type stateHandling string

const (
	hExact              stateHandling = "exact"                // rewound or restored per position
	hExactUnlessWrapped stateHandling = "exact unless wrapped" // exact until a ring wraps; then the path reports inexact itself
	hInexact            stateHandling = "inexact → cold"       // no per-position history: the path reports inexact and the caller cold-prefills
	hCleared            stateHandling = "cleared"              // zeroed or dropped
	hPersisted          stateHandling = "persisted"            // written by Snapshot and restored by LoadSession
	hRefused            stateHandling = "refused"              // the path declines while this kind is held (Snapshot → nil, rollback unsafe)
	hKept               stateHandling = "kept"                 // survives on purpose; the cell's reason says why that is right
	hTransient          stateHandling = "transient"            // built and dropped inside one forward or call; never live at this boundary
	hCallerBound        stateHandling = "caller-bound"         // not carried by this path; the caller must re-establish it (the reason names who does)
	hColdOnMismatch     stateHandling = "cold on mismatch"     // reuse only when it matches what the cached prefix was built under
)

// stateCell is one declaration: what the path does, and why that is correct. why is required for
// every handling except the self-evident exact/persisted/cleared ones.
type stateCell struct {
	h   stateHandling
	why string
}

// cacheStateGrid is the declaration. Every kind × every path has a cell.
var cacheStateGrid = map[cacheState]map[lifecyclePath]stateCell{
	statePositional: {
		lcRewind:       {hExact, ""},
		lcReset:        {hCleared, ""},
		lcSnapshot:     {hPersisted, ""},
		lcReuse:        {hExact, ""},
		lcSpecRollback: {hExact, ""},
	},
	stateRing: {
		lcRewind:       {hExactUnlessWrapped, "ring.truncate returns false for a wrapped ring rewound by more than one position (C1); TruncateTo passes it on"},
		lcReset:        {hCleared, ""},
		lcSnapshot:     {hPersisted, "the snapshot writes each ring's physical rows and count (C2)"},
		lcReuse:        {hExactUnlessWrapped, "rewindForReuse goes cold on an inexact TruncateTo"},
		lcSpecRollback: {hRefused, "specRollbackSafe refuses every SlidingWindow > 0 model: rollback sites do not consume an inexact rewind (audit C-04)"},
	},
	stateInt8KV: {
		lcRewind:       {hExact, ""},
		lcReset:        {hCleared, ""},
		lcSnapshot:     {hPersisted, ""},
		lcReuse:        {hExact, ""},
		lcSpecRollback: {hExact, ""},
	},
	stateMLALatent: {
		lcRewind:       {hExact, "TruncateTo re-slices the per-layer latent by latentDim"},
		lcReset:        {hCleared, ""},
		lcSnapshot:     {hRefused, "the snapshot format does not carry the latent store"},
		lcReuse:        {hExact, ""},
		lcSpecRollback: {hExact, ""},
	},
	stateDeltaNet:  recurrentCells,
	stateShortConv: recurrentCells,
	stateMamba2:    recurrentCells,
	stateKDA:       recurrentCells,
	stateImageBlocks: {
		lcRewind:       {hInexact, "the blocks are position ranges with no per-position rewind; a rewind below one would let new text attend bidirectionally"},
		lcReset:        {hCleared, "resetMultimodal (audit M-25)"},
		lcSnapshot:     {hRefused, "the snapshot format does not carry image blocks"},
		lcReuse:        {hInexact, "follows the rewind"},
		lcSpecRollback: {hTransient, "the speculative loops take text prompts; nothing they run puts an image block in their cache"},
	},
	stateMRoPE: {
		lcRewind:       {hInexact, "positions and the decode delta are set by the image prefill and have no per-position rewind"},
		lcReset:        {hCleared, "resetMultimodal (audit M-25)"},
		lcSnapshot:     {hRefused, "the snapshot format does not carry m-RoPE positions"},
		lcReuse:        {hInexact, "follows the rewind"},
		lcSpecRollback: {hTransient, "the speculative loops take text prompts; nothing they run sets m-RoPE positions in their cache"},
	},
	stateDeepstack: {
		lcRewind:       {hTransient, "GenerateQwenVL sets the rows for the image prefill and clears them right after (generate_vl.go)"},
		lcReset:        {hTransient, "cleared after the prefill that set them"},
		lcSnapshot:     {hTransient, "cleared after the prefill that set them"},
		lcReuse:        {hTransient, "cleared after the prefill that set them"},
		lcSpecRollback: {hTransient, "cleared after the prefill that set them"},
	},
	stateCapture: {
		lcRewind:       {hTransient, "captured rows are overwritten by every forward before anything reads them"},
		lcReset:        {hTransient, "overwritten by every forward"},
		lcSnapshot:     {hTransient, "overwritten by every forward"},
		lcReuse:        {hTransient, "overwritten by every forward"},
		lcSpecRollback: {hTransient, "overwritten by every forward"},
	},
	stateTree: {
		lcRewind:       {hTransient, "nothing in production sets the tree fields since EAGLE tree drafting was removed (be9aeea8, 2026-09-24); only tests do, on caches they own"},
		lcReset:        {hTransient, "unset in production (be9aeea8)"},
		lcSnapshot:     {hTransient, "unset in production (be9aeea8)"},
		lcReuse:        {hTransient, "unset in production (be9aeea8)"},
		lcSpecRollback: {hTransient, "unset in production (be9aeea8)"},
	},
	stateAdapter: {
		lcRewind:       {hKept, "the adapter is the stream's binding, not sequence state: positions are rewound under the same projections"},
		lcReset:        {hKept, "a reset session keeps its adapter; serve's per-adapter session LRUs rely on it"},
		lcSnapshot:     {hPersisted, "format v3 records the adapter name; LoadSession rebinds it, or refuses when the model has not loaded it"},
		lcReuse:        {hColdOnMismatch, "rewindForReuse goes cold when the bound adapter is not the one the cached prefix was built under"},
		lcSpecRollback: {hKept, "a rollback stays under the same projections"},
	},
}

// recurrentCells is the row every recurrent kind shares: state mutated in place per token, with no
// per-position history (hasRecurrentState's own comment).
var recurrentCells = map[lifecyclePath]stateCell{
	lcRewind:       {hInexact, "mutated in place per token; no per-position history to rewind to"},
	lcReset:        {hCleared, "resetRecurrent (audit C-01, C-03)"},
	lcSnapshot:     {hRefused, "the snapshot format does not carry recurrent state"},
	lcReuse:        {hInexact, "follows the rewind; rewindForReuse goes cold"},
	lcSpecRollback: {hRefused, "specRollbackSafe refuses every model with recurrent state"},
}

// kvCacheFieldState names the kind of state each KVCache field belongs to. Fields that are geometry,
// counters or scratch rather than per-sequence state are in kvCacheNotState instead, with a reason.
// TestKVCache_everyFieldHasAState fails a field that is in neither.
var kvCacheFieldState = map[string]cacheState{
	"keys": statePositional, "vals": statePositional, "stride": statePositional, "pos": statePositional, "manualPos": statePositional,
	"rings": stateRing, "localAny": stateRing,
	"quant": stateInt8KV, "keysQ": stateInt8KV, "valsQ": stateInt8KV, "keyScale": stateInt8KV, "valScale": stateInt8KV,
	"mlaLatent": stateMLALatent,
	"delta":     stateDeltaNet,
	"conv":      stateShortConv,
	"mamba":     stateMamba2,
	"kda":       stateKDA,
	"imgBlocks": stateImageBlocks,
	"mropePos":  stateMRoPE, "mropeDelta": stateMRoPE,
	"deepstack":     stateDeepstack,
	"captureLayers": stateCapture, "captured": stateCapture,
	"subCapture": stateCapture, "subAttn": stateCapture, "subMLP": stateCapture, "subMLPpre": stateCapture, "subCtx": stateCapture,
	"treeRowPos": stateTree, "treeMask": stateTree,
	"lora": stateAdapter,
}

var kvCacheNotState = map[string]string{
	"numLayers":       "geometry, fixed at NewCache",
	"kvDim":           "geometry, fixed at NewCache",
	"window":          "geometry: the ring capacity, fixed at NewCache",
	"headDim":         "geometry, fixed at NewCache",
	"latentDim":       "geometry: the MLA latent width, fixed at NewCache",
	"recurrentResets": "a counter of resetRecurrent calls, for tests (audit R-16)",
	"scr":             "per-stream scratch buffers, overwritten by every forward",
}

// holdsState reports whether c currently holds kind s. "Holds" follows allocation, as
// hasRecurrentState does: a recurrent kind is held from NewCache on, whether or not a token has run.
func (c *KVCache) holdsState(s cacheState) bool {
	switch s {
	case statePositional:
		return len(c.keys) > 0 || len(c.keysQ) > 0
	case stateRing:
		return c.localAny
	case stateInt8KV:
		return c.quant == kvI8
	case stateMLALatent:
		return len(c.mlaLatent) > 0
	case stateDeltaNet:
		return c.delta != nil
	case stateShortConv:
		return c.conv != nil
	case stateMamba2:
		return c.mamba != nil
	case stateKDA:
		return c.kda != nil
	case stateImageBlocks:
		return len(c.imgBlocks) > 0
	case stateMRoPE:
		return c.mropePos != nil || c.mropeDelta != 0
	case stateDeepstack:
		return c.deepstack != nil
	case stateCapture:
		return c.captureLayers != nil || c.subCapture
	case stateTree:
		return c.treeMask != nil || c.treeRowPos != nil
	case stateAdapter:
		return c.lora != nil
	}
	// An unknown kind is held: every caller asks "may I proceed", and the safe answer for a kind
	// this switch does not know is no. TestCacheStateGrid_holdsStateKnowsEveryKind keeps it unreached.
	return true
}

// holdsStateHandled reports whether c holds any kind whose cell on path p is h. It is how the
// lifecycle paths read the grid instead of hand-listing kinds.
func (c *KVCache) holdsStateHandled(p lifecyclePath, h stateHandling) bool {
	for _, s := range cacheStates {
		if cacheStateGrid[s][p].h == h && c.holdsState(s) {
			return true
		}
	}
	return false
}
