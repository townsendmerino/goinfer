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

import "maps"

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
		maps.Copy(out, m)
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
		pathCPUDecode:        ogTestedBy("TestDecodeParityInt4"),
		pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_cpuBatchedPrefill"),
		pathCPUBatchDecode:   ogTestedBy("TestOptionPath_cpuBatchedDecode"),
		pathSessionLifecycle: ogTestedBy("TestOptionPath_sessionSnapshot"),
		pathSpecVerify:       ogDeclinedAt("SpecDecodeConflict", "TestSpecDecodeConflict_refusesStagedWebGPUInt4"),
		pathResidentDecode:   ogTestedBy("TestOptionPathMetal_quant"),
		pathResidentPrefill:  ogTestedBy("TestOptionPathMetal_quant"),
	}, ogUntestedCell()),
	"EmbedInt4": ogFill(map[ogPath]ogCell{
		// TestOptionPath_cpuBatchedPrefill's sequential reference is this path (runLayers per token); a defect
		// planted in the decode-only embedding lookup turns it red.
		pathCPUDecode:        ogTestedBy("TestOptionPath_cpuBatchedPrefill"),
		pathCPUBatchDecode:   ogTestedBy("TestOptionPath_cpuBatchedDecode"),
		pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_cpuBatchedPrefill"),
		pathSpecVerify:       ogTestedBy("TestOptionPath_specVerify"),
		pathSessionLifecycle: ogNACell("a weight format; the session's cache does not depend on it"),
		// Metal's resident needs an int8 embedding table; an int4 one is declined at build and runs on the CPU.
		pathResidentDecode:  ogDeclinedAt("int8Buf", "TestOptionPathMetal_embedInt4Declines"),
		pathResidentPrefill: ogDeclinedAt("int8Buf", "TestOptionPathMetal_embedInt4Declines"),
	}, ogUntestedCell()),
	"ActQuantGroup": ogFill(map[ogPath]ogCell{
		pathCPUDecode: ogTestedBy("TestActQuantGroup_perModel"),
		// Tested on CUDA. Metal ignores the setting (2026-10-08): no code in metal/ reads it, and for a family
		// without the activation hazard residentAdmission admits the load, so a Metal resident runs per-vector
		// scales while the option asked for per-32 — the Options doc says other resident backends decline.
		// Recorded in docs/tasks/task-option-path-admission-2026-10.md; the resident prefill cell stays untested.
		pathResidentDecode:   ogTestedBy("TestActGroup_phi3ResidentMatchesCPU"),
		pathCPUBatchDecode:   ogTestedBy("TestOptionPath_cpuBatchedDecode"),
		pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_cpuBatchedPrefill"),
		pathSpecVerify:       ogTestedBy("TestOptionPath_specVerify"),
		pathSessionLifecycle: ogNACell("an activation-quantization setting; the session's cache does not depend on it"),
	}, ogUntestedCell()),

	"KVQuant": ogMerge(
		map[ogPath]ogCell{
			// TestOptionPath_cpuBatchedPrefill's sequential reference is this path; a defect planted in the decode-only
			// int8 attention branch turns it red. (TestKVI8_genParity sets m.kvI8 itself, not through Options.)
			pathCPUDecode:        ogTestedBy("TestOptionPath_cpuBatchedPrefill"),
			pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_cpuBatchedPrefill"), // TestKVI8_batchedPrefill sets m.kvI8 itself
			pathCPUBatchDecode:   ogDeclinedAt("cpuBatchCacheEligible", "TestCPUBatch_ineligibleCachesBypass"),
			pathSpecVerify:       ogTestedBy("TestOptionPath_specVerify"),
			pathSessionLifecycle: ogTestedBy("TestKVI8_snapshotRoundtrip"),
		},
		ogResidentNA("the CPU KV cache's precision (\"selects the CPU KV cache storage precision\"); a resident uses KVPrecision"),
	),
	"KVPrecision": ogMerge(
		ogCPUNA("\"Ignored off the residency path\""),
		map[ogPath]ogCell{
			pathResidentDecode:   ogTestedBy("TestOptionPathMetal_kvPrecision"),
			pathResidentPrefill:  ogDeclinedAt("PrefillPath", "TestPrefill_declinesInt8KV"),
			pathSpecVerify:       ogTestedBy("TestOptionPathMetal_kvPrecision"),
			pathSessionLifecycle: ogTestedBy("TestOptionPathMetal_kvPrecision"),
		},
	),

	// Untested on purpose (2026-10-08): the option's doc says bit-identical to fully resident, citing CUDA tests.
	// On Metal, with fewer slots than experts so experts are re-staged, mixtral-tiny at int4 decodes about 0.004
	// apart in log-probability from fully resident; nothing tests Metal's claim. With int8 experts Metal's build
	// panics (recovered into a decline). Recorded in docs/tasks/task-option-path-admission-2026-10.md.
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

	// The tested cells are expert paging on an MoE .giw. StreamWeights on a dense .giw streams whole layers
	// instead (layerPager); that branch is held by TestLayerPaging_bitExact, which needs a downloaded GGUF.
	"StreamWeights": ogMerge(
		ogResidentNA("CPU expert paging of an mmap-backed .giw (\"the CPU's expert paging is StreamWeights\")"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogTestedBy("TestOptionPath_moePaging"),
			pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_moePaging"),
			pathCPUBatchDecode:   ogDeclinedAt("cpuBatchModelEligible", "TestOptionPath_moePagingDeclinesCPUBatch"),
			pathSpecVerify:       ogTestedBy("TestOptionPath_moePaging"),
			pathSessionLifecycle: ogNACell("pages weights, not the session's cache"),
		},
	),
	"WeightCacheBytes": ogMerge(
		ogResidentNA("the budget of StreamWeights, a CPU paging option"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogTestedBy("TestOptionPath_moePaging"),
			pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_moePaging"),
			pathCPUBatchDecode:   ogDeclinedAt("cpuBatchModelEligible", "TestOptionPath_moePagingDeclinesCPUBatch"),
			pathSpecVerify:       ogTestedBy("TestOptionPath_moePaging"),
			pathSessionLifecycle: ogNACell("budgets weights, not the session's cache"),
		},
	),
	"MoEPager": ogMerge(
		ogResidentNA("the CPU expert pager's backing mode"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogTestedBy("TestOptionPath_moePaging"),
			pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_moePaging"),
			pathCPUBatchDecode:   ogDeclinedAt("cpuBatchModelEligible", "TestOptionPath_moePagingDeclinesCPUBatch"),
			pathSpecVerify:       ogTestedBy("TestOptionPath_moePaging"),
			pathSessionLifecycle: ogNACell("pages weights, not the session's cache"),
		},
	),

	"ResidentContext": ogMerge(
		ogCPUNA("\"Ignored off the residency path\""),
		map[ogPath]ogCell{
			pathResidentDecode:   ogTestedBy("TestOptionPathMetal_neutralOptions"),
			pathResidentPrefill:  ogTestedBy("TestOptionPathMetal_neutralOptions"),
			pathSpecVerify:       ogTestedBy("TestOptionPathMetal_neutralOptions"),
			pathSessionLifecycle: ogTestedBy("TestOptionPathMetal_neutralOptions"),
		},
	),
	"ResidentKVSlots": ogMerge(
		ogCPUNA("asks a GPU-resident backend for independent KV caches"),
		map[ogPath]ogCell{
			pathResidentDecode:   ogTestedBy("TestOptionPathMetal_neutralOptions"),
			pathResidentPrefill:  ogTestedBy("TestOptionPathMetal_neutralOptions"),
			pathSpecVerify:       ogTestedBy("TestOptionPathMetal_neutralOptions"),
			pathSessionLifecycle: ogTestedBy("TestOptionPathMetal_neutralOptions"),
		},
	),
	"ResidentPrefillChunk": ogMerge(
		ogCPUNA("chunks a resident's batched prefill under MC3"),
		map[ogPath]ogCell{
			pathResidentDecode:  ogNACell("chunks prefill; decode is one token"),
			pathResidentPrefill: ogTestedBy("TestOptionPathMetal_prefillChunk"), // at a chunk >= the batched-prefill floor; below it, see the test
			pathSpecVerify: ogNACell("chunking is mc3Prefill's, which only an MC3 holder's generation runs; the speculative " +
				"paths claim the resident exclusively (claimExclusive) and prefill whole"),
			pathSessionLifecycle: ogNACell("schedules a prefill; the session's cache is the same either way (chunk-invariant)"),
		},
	),
	"ExactPrefill": map[ogPath]ogCell{
		pathCPUDecode:        ogNACell("selects how a prompt is ingested; decode is one token"),
		pathCPUBatchPrefill:  ogTestedBy("TestOptionPath_cpuBatchedPrefill"),
		pathCPUBatchDecode:   ogNACell("selects how a prompt is ingested; decode is one token"),
		pathResidentDecode:   ogNACell("selects how a prompt is ingested; decode is one token"),
		pathResidentPrefill:  ogTestedBy("TestOptionPathMetal_exactPrefill"),
		pathSpecVerify:       ogTestedBy("TestOptionPath_specVerify"),
		pathSessionLifecycle: ogNACell("selects prefill numerics; a session's reuse rules do not depend on them"),
	},
	"CPUBatchDecode": ogMerge(
		ogResidentNA("chooses the CPU's MC3c batching; a resident batches through its own MC3"),
		map[ogPath]ogCell{
			pathCPUDecode:        ogNACell("selects whether decode joins a batch; a single generation is unaffected"),
			pathCPUBatchPrefill:  ogNACell("a decode setting"),
			pathCPUBatchDecode:   ogTestedBy("TestEnableCPUBatch_policy"),
			pathSpecVerify:       ogTestedBy("TestOptionPath_specVerify"),
			pathSessionLifecycle: ogTestedBy("TestOptionPath_sessionSnapshot"),
		},
	),
}

// optionGridUntestedCeiling is the ratchet: the number of ogUntested cells may not rise above it,
// and when it falls the constant must be lowered to match (TestOptionGrid_ratchet), so a cell
// that gains a test cannot quietly lose it again.
const optionGridUntestedCeiling = 9
