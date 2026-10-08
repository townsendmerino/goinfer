package decoder

// The option grid (docs/tasks/task-option-path-admission-2026-10.md §4.0, stage 2): every
// decoder.Options field against every execution path, with what that path does with it and the
// evidence. Step 1 (docs/measurements/audit-classes-2026-10-08.md) found 22 audit findings of one
// shape here — a load option no guard on a path knew about (-kv i8 into Metal's batched prefill,
// 09-30 A-C01; paged MoE slots into three Metal entry points, 09-10 C-08 and 09-12 C-02; int4 into
// WebGPU's speculative verify, 09-10 M-09).
//
// Owner decision 2026-10-08: no behaviour changes in this stage. A combination that runs today
// without a test that drives it is declared ogUntested and keeps running; TestOptionGrid_ratchet
// holds the number of such cells at a ceiling that can only come down. What fails closed is the
// registry itself: a new Options field fails TestOptionGrid_everyOptionClassified until it is
// classified on every path, and a cell that claims a test or a decline must name one that exists.

// ogPath is an execution path an option can reach.
type ogPath string

const (
	pathCPUDecode        ogPath = "CPU decode"                 // per-token forward on the CPU (and the sequential prefill)
	pathCPUBatchPrefill  ogPath = "CPU batched prefill"        // forwardN over a prompt (canBatchN)
	pathCPUBatchDecode   ogPath = "CPU batched decode"         // MC3c: concurrent generations joined into one forward (EnableCPUBatch)
	pathResidentDecode   ogPath = "GPU resident decode"        // CUDA / Metal / WebGPU resident per-token decode
	pathResidentPrefill  ogPath = "GPU resident prefill"       // a resident's batched prefill (PrefillPath on each resident)
	pathSpecVerify       ogPath = "speculative verify"         // the speculative loops' verify step (SpecDecodeConflict, specRollbackSafe)
	pathSessionLifecycle ogPath = "session reuse and snapshot" // Session prefix reuse, Snapshot / LoadSession
)

// ogPaths is every path, in display order.
var ogPaths = []ogPath{
	pathCPUDecode, pathCPUBatchPrefill, pathCPUBatchDecode,
	pathResidentDecode, pathResidentPrefill, pathSpecVerify, pathSessionLifecycle,
}

// ogHandling is what a path does with an option.
type ogHandling string

const (
	ogTested ogHandling = "tested" // honoured, and `test` drives the option through this path
	// "Drives" is strict (CLAUDE.md: a test that supplies its own calling convention vouches for the unit, not
	// for the system): the test sets the option through Options (or the public entry that sets it) and runs the
	// path. A test that sets the internal field itself shows the path works, not that the option reaches it.
	ogDeclined ogHandling = "declined"           // the path refuses the option at `decline`; `test` (optional) shows it does
	ogUntested ogHandling = "admitted, untested" // runs today with no test that drives it; counted by the ratchet
	ogNA       ogHandling = "n/a"                // the option does not reach this path; `why` says why
)

// ogCell is one declaration.
type ogCell struct {
	h       ogHandling
	test    string // a Test function in this repository (any module) that drives the cell
	decline string // the function where the path refuses the option
	why     string
}

func ogTestedBy(test string) ogCell { return ogCell{h: ogTested, test: test} }
func ogDeclinedAt(decline, test string) ogCell {
	return ogCell{h: ogDeclined, decline: decline, test: test}
}
func ogUntestedCell() ogCell           { return ogCell{h: ogUntested} }
func ogNACell(why string) ogCell       { return ogCell{h: ogNA, why: why} }
func ogAll(c ogCell) map[ogPath]ogCell { return ogFill(nil, c) }
func ogFill(m map[ogPath]ogCell, c ogCell) map[ogPath]ogCell {
	out := map[ogPath]ogCell{}
	for _, p := range ogPaths {
		if v, ok := m[p]; ok {
			out[p] = v
		} else {
			out[p] = c
		}
	}
	return out
}

// ogCPUNA / ogResidentNA mark an option's other side n/a, quoting the option's own documentation.
func ogCPUNA(why string) map[ogPath]ogCell {
	return map[ogPath]ogCell{pathCPUDecode: ogNACell(why), pathCPUBatchPrefill: ogNACell(why), pathCPUBatchDecode: ogNACell(why)}
}

func ogResidentNA(why string) map[ogPath]ogCell {
	return map[ogPath]ogCell{pathResidentDecode: ogNACell(why), pathResidentPrefill: ogNACell(why)}
}

func ogMerge(ms ...map[ogPath]ogCell) map[ogPath]ogCell {
	out := map[ogPath]ogCell{}
	for _, m := range ms {
		for p, c := range m {
			out[p] = c
		}
	}
	return out
}

// ogLoadOnly is an Options field consumed while loading (planning, pricing, refusing, merging) and
// read by no execution path afterwards. Every path is n/a for it, for the reason given.
var ogLoadOnly = map[string]string{
	"noSelfTest":                 "the resident self-test's own fixture loads; read by the self-test at load",
	"LoRA":                       "merged into the base weights at load (\"merged into the base at load\"), so every path runs the merged weights; compute-time adapters are Session state, in the cache-state grid",
	"AcceptSlowMoE":              "an acknowledgement checked when a paged-MoE load is planned; a no-op afterwards",
	"DisableFit":                 "selects load-time fit-by-default behaviour (resolveCtxCapFit, guardFit)",
	"ExtraResidentBytes":         "prices a companion allocation when the resident is planned",
	"ExtraResidentKVPerPosition": "prices a companion allocation's K/V when the resident context is chosen",
	"LoadAbort":                  "checked between layers during a GGUF weight build; nil afterwards in effect",
	"BackendAuto":                "tells a backend at load that \"auto\" chose it, so it may decline a precision it would requantize",
	"ResidentKVSlotsDefault":     "tells the backend at load that ResidentKVSlots is the caller's default, not a chosen count",
}

// ogGrid is the declaration for every Options field that reaches a path.
var ogGrid = map[string]map[ogPath]ogCell{
	"Backend": ogAll(ogNACell("selects which paths run; it is the row, not an input to one")),
	"Knobs": ogAll(ogNACell("routes per-model operator knobs by name to the code that reads each " +
		"(knobs.go); each knob is an environment read listed in testdata/env_reads.txt and owned there")),

	"Quant": ogFill(map[ogPath]ogCell{
		pathCPUDecode:       ogTestedBy("TestDecodeParityInt4"),
		pathCPUBatchPrefill: ogUntestedCell(), // TestInt4_forwardParity runs runLayers per token: decode, not forwardN
		pathSpecVerify:      ogDeclinedAt("SpecDecodeConflict", "TestSpecDecodeConflict_refusesStagedWebGPUInt4"),
	}, ogUntestedCell()),
	"EmbedInt4": ogFill(map[ogPath]ogCell{
		pathSessionLifecycle: ogNACell("a weight format; the session's cache does not depend on it"),
	}, ogUntestedCell()),
	"ActQuantGroup": ogFill(map[ogPath]ogCell{
		pathCPUDecode:        ogTestedBy("TestActQuantGroup_perModel"),
		pathResidentDecode:   ogTestedBy("TestActGroup_phi3ResidentMatchesCPU"),
		pathSessionLifecycle: ogNACell("an activation-quantization setting; the session's cache does not depend on it"),
	}, ogUntestedCell()),

	"KVQuant": ogMerge(
		map[ogPath]ogCell{
			pathCPUDecode:        ogUntestedCell(), // TestKVI8_genParity sets m.kvI8 itself, not through Options
			pathCPUBatchPrefill:  ogUntestedCell(), // TestKVI8_batchedPrefill likewise
			pathCPUBatchDecode:   ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogTestedBy("TestKVI8_snapshotRoundtrip"),
		},
		ogResidentNA("the CPU KV cache's precision (\"selects the CPU KV cache storage precision\"); a resident uses KVPrecision"),
	),
	"KVPrecision": ogMerge(
		ogCPUNA("\"Ignored off the residency path\""),
		map[ogPath]ogCell{
			pathResidentDecode:   ogUntestedCell(), // the Metal int8-KV parity tests drive kernels, not Options.KVPrecision
			pathResidentPrefill:  ogDeclinedAt("PrefillPath", "TestPrefill_declinesInt8KV"),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogUntestedCell(),
		},
	),

	"MoECacheExperts": ogMerge(
		ogCPUNA("\"CUDA and Metal residency; the CPU's expert paging is StreamWeights\""),
		map[ogPath]ogCell{
			pathResidentDecode:   ogUntestedCell(),
			pathResidentPrefill:  ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogUntestedCell(),
		},
	),
	"MoECacheSlots": ogMerge(
		ogCPUNA("\"Only meaningful with MoECacheExperts\", a residency option"),
		map[ogPath]ogCell{
			pathResidentDecode:   ogUntestedCell(),
			pathResidentPrefill:  ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogUntestedCell(),
		},
	),

	"StreamWeights": ogMerge(
		ogResidentNA("CPU expert paging of an mmap-backed .giw (\"the CPU's expert paging is StreamWeights\")"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogUntestedCell(),
			pathCPUBatchPrefill:  ogUntestedCell(),
			pathCPUBatchDecode:   ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogNACell("pages weights, not the session's cache"),
		},
	),
	"WeightCacheBytes": ogMerge(
		ogResidentNA("the budget of StreamWeights, a CPU paging option"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogUntestedCell(),
			pathCPUBatchPrefill:  ogUntestedCell(),
			pathCPUBatchDecode:   ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogNACell("budgets weights, not the session's cache"),
		},
	),
	"MoEPager": ogMerge(
		ogResidentNA("the CPU expert pager's backing mode"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogUntestedCell(),
			pathCPUBatchPrefill:  ogUntestedCell(),
			pathCPUBatchDecode:   ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogNACell("pages weights, not the session's cache"),
		},
	),

	"ResidentContext": ogMerge(
		ogCPUNA("\"Ignored off the residency path\""),
		map[ogPath]ogCell{
			pathResidentDecode:   ogUntestedCell(),
			pathResidentPrefill:  ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogUntestedCell(),
		},
	),
	"ResidentKVSlots": ogMerge(
		ogCPUNA("asks a GPU-resident backend for independent KV caches"),
		map[ogPath]ogCell{
			pathResidentDecode:   ogUntestedCell(),
			pathResidentPrefill:  ogUntestedCell(),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogUntestedCell(),
		},
	),
	"ResidentPrefillChunk": ogMerge(
		ogCPUNA("chunks a resident's batched prefill under MC3"),
		map[ogPath]ogCell{
			pathResidentDecode:   ogNACell("chunks prefill; decode is one token"),
			pathResidentPrefill:  ogUntestedCell(), // TestMC5_prefillChunkInvariance chunks PrefillLast itself; it never sets the option
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogNACell("schedules a prefill; the session's cache is the same either way (chunk-invariant)"),
		},
	),
	"ExactPrefill": map[ogPath]ogCell{
		pathCPUDecode:        ogNACell("selects how a prompt is ingested; decode is one token"),
		pathCPUBatchPrefill:  ogUntestedCell(),
		pathCPUBatchDecode:   ogNACell("selects how a prompt is ingested; decode is one token"),
		pathResidentDecode:   ogNACell("selects how a prompt is ingested; decode is one token"),
		pathResidentPrefill:  ogUntestedCell(),
		pathSpecVerify:       ogUntestedCell(),
		pathSessionLifecycle: ogNACell("selects prefill numerics; a session's reuse rules do not depend on them"),
	},
	"CPUBatchDecode": ogMerge(
		ogResidentNA("chooses the CPU's MC3c batching; a resident batches through its own MC3"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogNACell("selects whether decode joins a batch; a single generation is unaffected"),
			pathCPUBatchPrefill:  ogNACell("a decode setting"),
			pathCPUBatchDecode:   ogTestedBy("TestEnableCPUBatch_policy"),
			pathSpecVerify:       ogUntestedCell(),
			pathSessionLifecycle: ogUntestedCell(),
		},
	),
}

// optionGridUntestedCeiling is the ratchet: the number of ogUntested cells may not rise above it,
// and when it falls the constant must be lowered to match (TestOptionGrid_ratchet), so a cell
// that gains a test cannot quietly lose it again.
const optionGridUntestedCeiling = 57
