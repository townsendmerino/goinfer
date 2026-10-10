package main

import (
	"bufio"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// The parity sweep — the release-tag ritual. One full, asset-gated parity sweep on the EXACT commit
// you intend to tag, so every family/quant/tokenizer gate is green on the release SHA and not merely
// on some commit along the way.
//
// THIS IS A CHECKSET, NOT A TALLY, and that is the whole reason it needs its own decision rather
// than heavy's. `gate heavy` asks "did anything fail?"; the sweep asks, of each NAMED gate, "is this
// one green?" — which makes a gate that never ran a distinct and much worse outcome than a gate that
// failed, because a tally cannot see it at all. Four outcomes, three of them blocking:
//
//	PASS     ✅
//	FAIL     a blocker — UNLESS the ledger says FIRST-RUN, i.e. it failed with no confirmed prior
//	         result, so there is no second point to compute a delta from. Reported as an ITEM.
//	SKIP     a blocker (the asset exists somewhere and this box is mis-provisioned) — UNLESS the
//	         gate is in assetNeverBuilt, where no invocation on any machine could make it green, so
//	         it is a COVERAGE GAP. A permanent blocker is not a gate; it is an override habit.
//	MISSING  the test produced no result at all — a blocker, and the one a tally would miss.

type gateCheck struct{ Family, Test string }

// The required gates: one or two canonical gates per family, plus the quant-format and tokenizer
// parity gates. Each MUST report PASS.
var parityGates = []gateCheck{
	{"gemma3", "TestGGUF_gemma3_parity"},
	{"gemma3-forward", "TestForward_logitParity"},
	{"gemma3-sliding", "TestForward_slidingWindowParity"},
	{"gemma4-E2B/E4B", "TestGemma4_logitParity"},
	{"gemma4-12B", "TestGemma4_12B_logitParity"},
	{"qwen2", "TestQwen2_forwardParity"},
	{"qwen2-gguf", "TestGGUF_qwen2_parity"},
	{"qwen3", "TestQwen3_forwardParity"},
	{"qwen3_asr-text", "TestQwen3ASR_tinyTextParity"},               // Qwen3-ASR's decoder from the checkpoint's own layout, against transformers
	{"qwen3_asr-composition", "TestQwen3ASR_tinyCompositionParity"}, // audio samples -> front end -> encoder -> splice -> decoder, against transformers
	{"voxtral-text", "TestVoxtral_tinyTextParity"},                  // Voxtral's Llama decoder from the checkpoint's own layout (language_model.*, untied head, head_dim != hidden/heads), against transformers
	{"voxtral-composition", "TestVoxtral_tinyCompositionParity"},    // audio samples -> whole-signal front end -> tower -> stacking -> projector -> splice -> decoder, on 1 and 2 windows, against transformers
	{"qwen3-gguf", "TestGGUF_qwen3_parity"},
	{"qwen2moe", "TestQwen2Moe_forwardParity"},
	{"qwen3moe", "TestQwen3Moe_forwardParity"},
	{"granite-dense", "TestGraniteDense_forwardParity"},
	{"mistral3", "TestMinistral3_forwardParity"},
	{"smollm3", "TestSmolLM3_forwardParity"},
	{"olmo3", "TestOlmo3_forwardParity"},
	{"gemma+gemma2", "TestGemma12_forwardParity"}, // Gemma 1 / CodeGemma and Gemma 2, safetensors and GGUF
	{"olmo_hybrid", "TestOlmoHybrid_forwardParity"},
	{"bailing_hybrid", "TestBailingHybrid_forwardParity"},
	{"mistral3-batched", "TestMinistral3_batchedMatchesSequential"},
	{"llama", "TestLlama_forwardParity"},
	{"llama3.2", "TestLlama32_forwardParity"},
	{"mistral", "TestMistral_forwardParity"},
	{"mixtral", "TestMixtral_forwardParity"},
	{"mellum2", "TestMellum2_logitParity"},
	{"mellum2-window", "TestMellum2_windowParity"},
	{"gpt2", "TestGPT2_forwardParity"},
	{"qwen3_5_moe-tiny", "TestQwen35_forwardParity"},
	{"deltanet", "TestGatedDeltaNet_parity"},
	{"deepseek-tiny", "TestDeepseek_textParity"},
	{"kimi-tiny", "TestKimi_textParity"},
	{"phi3-tiny", "TestPhi3_textParity"},
	{"llama4-tiny", "TestLlama4_textParity"},
	{"nemotron-tiny", "TestNemotron_textParity"},
	{"nemotron3nano-tiny", "TestNemotron3NanoMoE_textParity"},
	{"gguf-Q8_0", "TestGGUF_Q8_0_parity"},
	{"gguf-Q4_0", "TestGGUF_Q4_0_parity"},
	{"gguf-Q4_K_M", "TestGGUF_Q4_K_M_parity"},
	{"gguf-Q4_K_S", "TestGGUF_Q4_K_S_parity"},
	{"gguf-Q5_K_M", "TestGGUF_Q5_K_M_parity"},
	{"gguf-Q6_K", "TestGGUF_Q6_K_parity"},
	{"gguf-Q3_K_M", "TestGGUF_Q3_K_M_parity"},
	{"gguf-Q2_K", "TestGGUF_Q2_K_parity"},
	{"gptq", "TestGPTQ_parity"},
	{"awq", "TestAWQ_parity"},
	{"w4a8-int4", "TestW4A8DecodeParity"},
	// A required gate that is not named here is a gate whose failure nobody can read: the int4 forward gate,
	// the broadest quant check, surfaces as an anonymous "1 fail" without its entry.
	{"int4-forward", "TestInt4_forwardParity"},
	{"tok-gemma", "TestEncodeDecode_goldenParity"},
	{"tok-qwen3", "TestByteLevel_qwen3GoldenParity"},
	{"tok-llama3", "TestByteLevel_llama3GoldenParity"},
	{"tok-mellum2", "TestByteLevel_mellum2GoldenParity"},
	{"tok-tinyllama", "TestLoadGGUF_tinyllamaParity"},
}

// Real-checkpoint gates (build tag realckpt). Heavy — large downloads and RAM; each SKIPs when its
// asset is absent. Run unless REALCKPT=0.
var parityRealckptGates = []gateCheck{
	{"qwen3.6-gguf-gate", "TestQwen35GGUF_gate"},
	{"qwen3.6-weightdiff", "TestQwen35GGUF_weightDiff"},
	{"qwen3.6-gate2", "TestQwen35Real_gate2FullModel"},
	{"deepseek-v2lite", "TestDeepseekV2LiteReal_gate"},
	{"deepseek-moonlight", "TestDeepseekMoonlightReal_gate"},
	{"deepseek-gguf", "TestDeepseekGGUFReal_gate"},
	{"phi3-mini-oracle", "TestPhi3MiniReal_gate"},
	{"phi3-gguf", "TestPhi3GGUFReal_gate"},
	{"llama4-scout-gguf", "TestLlama4Real_gate"},
	{"qwen3next-oracle", "TestQwen3NextReal_oracle"},
	{"nemotron35lightning-oracle", "TestNemotron35LightningReal_oracle"},
	{"qwen3moe-oracle", "TestQwen3MoeReal_oracle"},
	// One or two canonical gates for families with no tiny fixture, so no gate in parityGates: "not required"
	// is the absence of a decision, and the sweep's own report could not tell the two apart. Every asset
	// below is registered in testdata/assets.json, so none is a permanent SKIP-blocker.
	{"gpt_oss", "TestGptOssReal_gate"},
	{"gpt_oss-logits", "TestGptOssReal_logitParity"},
	{"granite-gguf", "TestGraniteReal_gate"},
	{"granite-oracle", "TestGraniteReal_oracle"},
	{"laguna", "TestLagunaReal_gate"},
	{"laguna-oracle", "TestLagunaReal_oracle"},
	{"qwen25vl-real", "TestQwen25VLReal_gate"},
	{"qwen3vl-real", "TestQwen3VLReal_gate"}, // Qwen3-VL-2B text forward against HF f32 (the image prompt is TestQwen3VLImageReal, which needs its own pin output)
	{"laguna-gguf", "TestLagunaGGUF_gate"},
	{"glm4moe-air", "TestGlm4MoeAir_gate"},
	{"cohere", "TestCohereAyaReal_gate"},
	{"cohere2", "TestCohere2R7bReal_gate"},
	{"qwen3.8-dense", "TestQwen38Real_gate"},
	{"qwen3.8-dense-oracle", "TestQwen38Real_oracle"},
	{"qwen3.8-gguf", "TestQwen38GGUF_gate"},
	{"qwen3.8-gguf-weightdiff", "TestQwen38GGUF_weightDiff"},
	// Register a real-checkpoint gate here in the SAME change that adds its asset and test:
	// TestRealckptGateIsListedOrExplicitlyNotRequired fails otherwise.
	{"smollm3-3b", "TestSmolLM3_3bReal_gate"},
	{"lfm2-2.6b", "TestLFM2Real_gate"},
	{"mistral3-real", "TestMinistral3Real_gate"},
	{"spark2_5-1.7b", "TestSpark25Real_gate"},
	// granite-dense-real is deliberately NOT named "granite-*" alone — granite-gguf/granite-oracle
	// above already claim that shape for the granitemoehybrid family; the label makes the two
	// unmistakable in the sweep's own report, not just in code.
	{"granite-dense-real", "TestGraniteDenseReal_gate"},
	{"olmo3-oracle", "TestOlmo3Real_gate"},
	{"gemma12-oracle", "TestGemma12Real_gate"},
	{"olmo-hybrid-oracle", "TestOlmoHybridReal_gate"},
	{"internlm2-real", "TestInternLM2_1_8bReal_gate"},
	{"qwen2moe-oracle", "TestQwen2MoeReal_oracle"},
}

// Not listed here or in realckptNotRequired: cuda/qwen25vl_resident_real_test.go's
// TestQwen25VLResidentReal_gate lives in package cuda (it needs the cuda backend's init(), and decoder
// cannot import cuda) under `cuda && goinfer_testhooks`, not `realckpt`, so this list's discipline does
// not reach it, as with its siblings in that package.

// emitGates are the numeric-oracle gates expected to record a manifest row under EMIT_MANIFEST. Family is
// the manifest family the gate writes, so the pair is the other way round from the lists above.
var emitGates = []gateCheck{
	{"phi3", "TestPhi3MiniReal_gate"},
	{"deepseek_v2", "TestDeepseekV2LiteReal_gate"},
	{"deepseek_v3", "TestDeepseekMoonlightReal_gate"},
	{"qwen3_5_moe", "TestQwen35Real_gate2FullModel"},
	{"cohere", "TestCohereAyaReal_gate"},
	{"cohere2", "TestCohere2R7bReal_gate"},
	{"qwen3_next", "TestQwen3NextReal_oracle"},
	{"qwen3_moe", "TestQwen3MoeReal_oracle"},
	{"lfm2", "TestLFM2Real_gate"},
	{"mistral3", "TestMinistral3Real_gate"},
	{"granite", "TestGraniteDenseReal_gate"},
	{"olmo3", "TestOlmo3Real_gate"},
	{"gemma+gemma2", "TestGemma12Real_gate"}, // gemma-2b-it, codegemma-2b, gemma-2-2b-it: safetensors f32 + llama.cpp Q8_0 GGUF
	{"olmo_hybrid", "TestOlmoHybridReal_gate"},
	{"internlm2", "TestInternLM2_1_8bReal_gate"},
	{"qwen2_moe", "TestQwen2MoeReal_oracle"},
	{"laguna", "TestLagunaReal_oracle"},
	{"qwen3_5", "TestQwen38Real_oracle"},
	{"qwen2_5_vl", "TestQwen25VLReal_gate"},
	{"qwen3_vl", "TestQwen3VLReal_gate"}, // Qwen3-VL-2B text forward against HF f32; the image prompt is TestQwen3VLImageReal (needs scripts/pin_qwen3vl_image_real.py output, so not a required gate)
}

// assetNeverBuilt names required gates whose asset has NEVER been built anywhere, so no invocation can
// make them green. They are reported and counted as coverage gaps, not blockers.
//
// The list is empty, which is the honest state; the machinery stays for the next gate whose asset has
// never been built. The only correct way OFF the list is to build the asset.
var assetNeverBuilt = map[string]bool{}

// gateRunFilter reads GATE_RUN, an optional narrowing filter for a granular re-run of the sweep (e.g.
// after fixing one blocker) that keeps the same checkset classification, ledger handling and verdict
// logic a bare `go test -run` would skip. "" (the default) means the full sweep.
func gateRunFilter() (re *regexp.Regexp, raw string) {
	raw = env("GATE_RUN", "")
	if raw == "" {
		return nil, ""
	}
	re, err := regexp.Compile(raw)
	if err != nil {
		// Fail LOUD, not silently back to "everything": a bad GATE_RUN that quietly ran the full
		// sweep would look identical to a correctly-scoped one until someone timed it.
		fmt.Fprintf(os.Stderr, "!! GATE_RUN=%q does not compile as a regexp: %v — refusing rather than "+
			"silently running the full sweep\n", raw, err)
		os.Exit(2)
	}
	return re, raw
}

// scopeChecks narrows a required checkset to the entries a GATE_RUN filter selects.
//
// It narrows the required list itself, not just which tests execute: otherwise every gate outside the
// filter would report "DID NOT RUN (blocker)" and a scoped re-run would look like a broken sweep. The
// tradeoff: a scoped run's "ALL REQUIRED GATES GREEN" answers "is everything I selected green", not "is
// the release ready", and never substitutes for the full unfiltered sweep before a tag.
func scopeChecks(checks []gateCheck, filter *regexp.Regexp) []gateCheck {
	var scoped []gateCheck
	for _, g := range checks {
		if filter.MatchString(g.Test) {
			scoped = append(scoped, g)
		}
	}
	return scoped
}

// parityCells builds the sweep's two cells. env carries whatever the asset preflight resolved.
func parityCells(env map[string]string, realckpt bool, timeout string) []cell {
	_, rawFilter := gateRunFilter()
	base := cell{
		Name:    "./decoder/ ./tokenizer/",
		Pkgs:    []string{"./decoder/", "./tokenizer/"},
		Timeout: timeout,
		Env:     env,
		// GATE_RUN narrows the base cell the same way it narrows realckpt below — applied directly
		// since this cell has no pre-derived name list to filter, unlike realckptRun's.
		Run: rawFilter,
	}
	cells := []cell{base}
	if realckpt {
		run, _ := realckptRun()
		cells = append(cells, cell{
			Name: "realckpt real-model gates",
			Pkgs: []string{"./decoder/"},
			Tags: []string{"realckpt"},
			// DERIVED FROM THE TAGGED FILES (realckptRun), not hand-written: a -run filter that cannot reach a gate is
			// not a skip, it is a gate that silently does not exist. The cell is still filtered because the realckpt
			// tag also carries perf and diagnostic tests the release sweep is not for; the filter selects on SHAPE,
			// not on a name list.
			Run:     run,
			Timeout: timeout,
			Env:     env,
		})
	}
	return cells
}

// assetPreflight resolves the asset environment from the SHARED REGISTRY (testdata/assets.json) and
// reports what it resolved. The gates skip-if-absent and a skip is reported as a blocker, so an unset
// variable and a genuinely missing checkpoint would be indistinguishable in the output; the registry is
// the single implementation of "is this asset present" (a bare path-exists test is satisfied by a
// directory where the loader wants the file inside).
//
// A preflight that does not run is announced LOUDLY rather than left to become a blocker cascade: without
// python3 every asset-gated gate skips and the count is about the failure, not the tree.
func assetPreflight(w io.Writer) map[string]string {
	env := map[string]string{
		// SET, not required. This is the release sweep; loading multi-GB checkpoints is its entire
		// purpose, and making the operator remember an opt-in whose absence reads as "asset missing
		// (blocker)" is a trap, not a safety feature.
		"GOINFER_HEAVY_TESTS": "1",
	}
	tmp, err := os.CreateTemp("", "gate_assets.*.env")
	if err != nil {
		fmt.Fprintf(w, "   !! could not create the preflight export file: %v\n", err)
		return env
	}
	path := tmp.Name()
	tmp.Close()
	defer os.Remove(path)

	cmd := exec.Command("python3", "scripts/asset_registry.py", "preflight", "--export-to", path)
	cmd.Stdout, cmd.Stderr = w, w
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(w, "   !! ASSET PREFLIGHT DID NOT RUN (scripts/asset_registry.py failed above: %v).\n", err)
		fmt.Fprintf(w, "   !! No asset variable has been resolved. Every asset-gated gate below will skip and be\n")
		fmt.Fprintf(w, "   !! counted as a blocker. THAT COUNT IS ABOUT THIS FAILURE, NOT ABOUT THE TREE.\n")
		return env
	}
	maps.Copy(env, parseExports(path))
	fmt.Fprintf(w, "   GOINFER_HEAVY_TESTS=1 set by this gate — the release sweep loads real checkpoints\n")
	return env
}

// parseExports reads `export NAME="value"` lines, the format asset_registry.py writes for the shell
// to source.
func parseExports(path string) map[string]string {
	out := map[string]string{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		line = strings.TrimPrefix(line, "export ")
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		out[strings.TrimSpace(k)] = strings.Trim(strings.TrimSpace(v), `"`)
	}
	return out
}

// ledgerClassify asks the gate ledger whether a FAILING gate has a confirmed prior result.
// Unreachable ledger → CONFIRMED, i.e. treat the failure as a blocker: the fail-SAFE direction, since
// the alternative would let a broken ledger silently downgrade every regression to an item.
func ledgerClassify(name string) string {
	root, err := repoRoot()
	if err != nil {
		return "CONFIRMED"
	}
	return classifyGate(root, name)
}

// runParity is the sweep's decision. Returns the process exit code.
func runParity(w io.Writer, logDir string) int {
	realckpt := env("REALCKPT", "1") == "1"
	emitManifest := env("EMIT_MANIFEST", "0") == "1"
	timeout := env("TIMEOUT", "180m")

	cellEnv := assetPreflight(w)
	if emitManifest {
		cellEnv["GOINFER_MANIFEST_EMIT"] = "1"
		fmt.Fprintf(w, "\n== EMIT_MANIFEST=1: real-oracle gates will record measured rows into the manifest ==\n")
	}

	prov := gatherProvenance([][2]string{
		{"realckpt", fmt.Sprintf("%v", realckpt)},
		{"timeout", timeout},
		{"emit", fmt.Sprintf("%v", emitManifest)},
	})
	fmt.Fprintf(w, "\n%s== parity sweep provenance ==%s\n", bold, off)
	prov.write(w)

	// THE COMPOSITION, NOT JUST THE VERDICT. This gate's axes are family × quant × loader, and a pass COUNT
	// alone cannot distinguish "the axes are covered" from "an axis collapsed to one value".
	composition(w, false)

	// RCIsFailure: a `go test` that exits non-zero with ZERO --- FAIL lines (a panic in a goroutine, a fatal
	// error, a timeout, a build failure) is a red cell. Otherwise a crashed realckpt cell produces no
	// named-gate result and the sweep reads that as gates it had nothing to say about.
	cfg := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure: true}
	cfg.Cells = parityCells(cellEnv, realckpt, timeout)
	if realckpt {
		if _, note := realckptRun(); note != "" {
			fmt.Fprintf(w, "   %s\n", note)
		}
	}
	res := newResults()
	var cells []cellResult
	for _, c := range cfg.Cells {
		fmt.Fprintf(w, "-- running %s ...\n", c.Name)
		cr := runCell(c, cfg, res, logDir)
		cells = append(cells, cr)
		fmt.Fprintf(w, "   %d pass / %d skip / %d fail (go rc %d, %s)  log: %s\n",
			cr.Pass, cr.Skip, cr.Fail, cr.RC, cr.Dur.Round(1e9), cr.LogPath)
	}

	checks := parityGates
	if realckpt {
		checks = append(append([]gateCheck{}, parityGates...), parityRealckptGates...)
	}
	if runFilter, raw := gateRunFilter(); runFilter != nil {
		scoped := scopeChecks(checks, runFilter)
		fmt.Fprintf(w, "\n== GATE_RUN=%q: checkset narrowed from %d to %d required gate(s) — "+
			"THIS IS A SCOPED RE-RUN, NOT A RELEASE-READY VERDICT ==\n", raw, len(checks), len(scoped))
		checks = scoped
	}

	rows, blockers, gaps, firstRuns := classifyChecks(res, checks, ledgerClassify, cfg.Cells)
	var checked []string
	fmt.Fprintf(w, "\n%-20s %-34s %s\n", "FAMILY", "GATE", "RESULT")
	fmt.Fprintln(w, strings.Repeat("-", 72))
	for _, r := range rows {
		checked = append(checked, r.Test)
		fmt.Fprintf(w, "%-20s %-34s %s\n", r.Family, r.Test, r.Mark)
	}

	// Safety net: any OTHER parity/gate-shaped test that skipped — a family the lists forgot.
	fmt.Fprintf(w, "\n-- catch-all: other *Parity / *gate tests that SKIPPED (review) --\n")
	for _, name := range catchAllSkips(res, checks) {
		fmt.Fprintf(w, "   skipped: %s\n", name)
	}

	blockers += extraBlockers(w, res, checks, cells, realckpt)

	if emitManifest {
		blockers += mergeManifest(w, res, logDir)
	}

	fmt.Fprintln(w)
	if firstRuns > 0 {
		fmt.Fprintf(w, "== %d FIRST-RUN: failed with no confirmed prior result ==\n", firstRuns)
		fmt.Fprintf(w, "   Reported, NOT counted as blockers -- there is no second point to compute a delta from.\n")
		fmt.Fprintf(w, "   NOT a claim they are harmless: each is an ITEM. Confirm a value deliberately with\n")
		fmt.Fprintf(w, "   go run ./cmd/gate ledger promote --gate <G> --value <V> --by <you>.\n")
	}
	if root, err := repoRoot(); err == nil {
		reconcileLedger(w, root, checked) // advisory, exactly as the shell had it
	}
	if gaps > 0 {
		fmt.Fprintf(w, "== %d COVERAGE GAP(S): required gates whose asset has never been built ==\n", gaps)
		fmt.Fprintf(w, "   Reported, not counted as blockers. No invocation can clear them -- only building the\n")
		fmt.Fprintf(w, "   asset can. A permanent blocker is not a gate, it is an override habit.\n")
	}
	verdict := "ALL REQUIRED GATES GREEN"
	if blockers > 0 {
		verdict = fmt.Sprintf("%d BLOCKER(S)", blockers)
	}
	extra := ""
	if gaps > 0 {
		extra += fmt.Sprintf(" (+%d coverage gap)", gaps)
	}
	if firstRuns > 0 {
		extra += fmt.Sprintf(" (+%d first-run)", firstRuns)
	}
	fmt.Fprintf(w, "== %s: %s%s ==\n", prov.Commit, verdict, extra)
	for _, c := range cells {
		fmt.Fprintf(w, "full log: %s\n", c.LogPath)
	}
	if blockers > 0 {
		return 1
	}
	return 0
}

// checkRow is one named gate's outcome, ready to print.
type checkRow struct{ Family, Test, Mark string }

// classifyChecks applies the sweep's four-outcome decision to the named gates. Split out from
// runParity so the decision — which is the entire gate — is mutation-testable without a real sweep.
// ledger is injected for the same reason.
func classifyChecks(res *results, checks []gateCheck, ledger func(string) string, cells []cell) (rows []checkRow, blockers, gaps, firstRuns int) {
	for _, g := range checks {
		act, seen := res.lookupTop(g.Test)
		var mark string
		switch {
		case !seen:
			// THE OUTCOME A TALLY CANNOT SEE. A required gate that produced no result at all — a
			// renamed test, a -run filter that stopped matching, a package that failed to build —
			// leaves a pass/skip/fail count entirely undisturbed. It is a blocker.
			mark = "⛔ DID NOT RUN (blocker) — " + whyNoResult(g.Test, cells)
			blockers++
		case act == "pass":
			mark = "✅ pass"
		case act == "fail" && neverConfirmed[g.Test] != "":
			// neverConfirmed's own promise ("its failure is reported as an ITEM and never blocks a tag") is checked
			// FIRST, so a live sweep enforces it and not only the static TestParity_everyRequiredGateIsConfirmed.
			mark = "FAIL - but in neverConfirmed (NOT a blocker): " + neverConfirmed[g.Test]
			gaps++
		case act == "fail":
			// THE FOURTH OUTCOME (B14). A gate failing with no confirmed prior result is asserting a
			// DELTA IT HAS NO SECOND POINT TO COMPUTE, so it is an ITEM, not a blocker — the change
			// that made a failure visible is the one change that provably did not cause it. This is
			// about ATTRIBUTION, not harmlessness: the first observed value must not be banked as a
			// baseline without a person deciding it is correct (`gate ledger promote`).
			switch ledger(g.Test) {
			case "FIRST-RUN":
				mark = "FIRST-RUN - failed, no confirmed prior result (ITEM, not a blocker)"
				firstRuns++
			case "SOURCE-CHANGED":
				mark = "FAIL (blocker) - NOTE: confirmed before this gate last changed"
				blockers++
			default:
				mark = "FAIL (blocker)"
				blockers++
			}
		case act == "skip":
			switch {
			case assetNeverBuilt[g.Test]:
				mark = "COVERAGE GAP - asset never built (NOT a blocker; see assetNeverBuilt)"
				gaps++
			case neverConfirmed[g.Test] != "":
				// neverConfirmed's promise ("never blocks a tag") covers a SKIP as well as a FAIL: an entry whose asset is
				// absent on this box would otherwise block every run, and the deferral would be written down but never
				// take effect.
				mark = "SKIP - asset missing, but in neverConfirmed (NOT a blocker): " + neverConfirmed[g.Test]
				gaps++
			default:
				mark = "SKIP - asset missing (blocker)"
				blockers++
			}
		}
		rows = append(rows, checkRow{g.Family, g.Test, mark})
	}
	return rows, blockers, gaps, firstRuns
}

// catchAllSkips lists top-level skipped tests whose names look like a parity/gate/golden check and
// are not already in the checkset — the "a family we forgot" net.
func catchAllSkips(res *results, checks []gateCheck) []string {
	seen := map[string]bool{}
	var out []string
	for _, k := range res.order {
		if isSubtest(k) || res.final[k] != "skip" || !strings.HasPrefix(k.Test, "Test") {
			continue
		}
		low := strings.ToLower(k.Test)
		if !strings.Contains(low, "parity") && !strings.Contains(low, "gate") && !strings.Contains(low, "golden") {
			continue
		}
		// Substring containment, not equality: the shell excluded by grepping the whole result line
		// against the gate names, so a test whose name CONTAINS a checkset name was excluded too.
		skip := false
		for _, g := range checks {
			if strings.Contains(k.Test, g.Test) {
				skip = true
				break
			}
		}
		if skip || seen[k.Test] {
			continue
		}
		seen[k.Test] = true
		out = append(out, k.Test)
	}
	sort.Strings(out)
	return out
}

// mergeManifest folds the collected PARITY_ROW lines into testdata/parity_manifest.json, re-renders
// the capability matrix, and reports emitter coverage. Returns how many blockers it added.
//
// The emitter only writes a row for a gate that PASSED (its t.Failed guard), so this never records a
// failed or skipped gate.
func mergeManifest(w io.Writer, res *results, logDir string) int {
	fmt.Fprintf(w, "\n-- EMIT_MANIFEST: merging measured PARITY_ROW lines into testdata/parity_manifest.json --\n")
	rows := res.parityRows
	if len(rows) == 0 {
		fmt.Fprintf(w, "   no PARITY_ROW lines emitted (no numeric-oracle gate ran with its asset)\n")
		emitterCoverage(w, res, nil)
		return 0
	}
	path := filepath.Join(logDir, "gate_parity_rows.txt")
	if err := os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		fmt.Fprintf(w, "   ❌ could not write the collected rows: %v\n", err)
		return 1
	}
	fams := map[string]bool{}
	for _, r := range rows {
		if f := jsonField(r, "family"); f != "" {
			fams[f] = true
		}
	}
	blockers := 0
	merge := exec.Command("go", "test", "./decoder", "-run", "^TestParityManifest_merge$", "-merge-rows", path, "-count=1")
	if out, err := merge.CombinedOutput(); err != nil {
		fmt.Fprintf(w, "   ❌ merge failed: %v\n%s\n", err, trunc(string(out), 2000))
		blockers++
	} else {
		fmt.Fprintf(w, "   merged %d row(s); families recorded: %s\n", len(rows), strings.Join(sortedSet(fams), " "))
		fmt.Fprintf(w, "-- re-rendering docs/capability-matrix.{md,json} from the manifest --\n")
		rr := exec.Command("go", "test", "./decoder", "-run", "CapabilityMatrix|ParityManifest", "-update", "-count=1")
		if out, err := rr.CombinedOutput(); err != nil {
			fmt.Fprintf(w, "   ❌ matrix -update failed: %v\n%s\n", err, trunc(string(out), 2000))
			blockers++
		} else {
			fmt.Fprintf(w, "   matrix + manifest re-rendered\n")
		}
	}
	emitterCoverage(w, res, fams)
	return blockers
}

// emitterCoverage flags a numeric-oracle gate that PASSED but recorded no row — a missing
// emitParityRow call, which is invisible in a pass count.
func emitterCoverage(w io.Writer, res *results, fams map[string]bool) {
	fmt.Fprintf(w, "-- emitter coverage (numeric-oracle gates) --\n")
	for _, g := range emitGates {
		act, seen := res.lookupTop(g.Test)
		// A gate that records several families names them joined by "+" ("gemma+gemma2"); the merged-row set holds
		// each family on its own, so look the parts up one by one. Looking the joined string up never matched, and
		// the v0.21.0 sweep warned "emitted NO row for gemma+gemma2" while both rows had been merged.
		var missing []string
		for f := range strings.SplitSeq(g.Family, "+") {
			if !fams[f] {
				missing = append(missing, f)
			}
		}
		switch {
		case len(missing) == 0:
			fmt.Fprintf(w, "   %-34s ✅ recorded row (%s)\n", g.Test, g.Family)
		case seen && act == "pass":
			fmt.Fprintf(w, "   %-34s ⚠️  PASSED but emitted NO row for %s (emitter missing?)\n", g.Test, strings.Join(missing, "+"))
		default:
			r := "MISSING"
			if seen {
				r = strings.ToUpper(act)
			}
			fmt.Fprintf(w, "   %-34s — %s (no row expected)\n", g.Test, r)
		}
	}
}

// jsonField pulls one top-level string field out of a compact PARITY_ROW payload without decoding
// the whole thing — the rows are written by emitParityRow with json.Marshal'd values.
func jsonField(line, key string) string {
	i := strings.Index(line, `"`+key+`":"`)
	if i < 0 {
		return ""
	}
	rest := line[i+len(key)+4:]
	before, _, ok := strings.Cut(rest, `"`)
	if !ok {
		return ""
	}
	return before
}

func sortedSet(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// whyNoResult distinguishes the two causes of "no result" that look identical in a report and are fixed in
// completely different places. A gate no pattern selects cannot be fixed by any machine, asset or
// environment; a gate that IS selected and still produced nothing is a different problem. Saying which
// halves the search.
//
// KNOWN LIMITATION: it checks -run selection only, not build-tag reachability. parityCells' base cell has
// an empty Run and no Tags, so the first branch fires for every test name at the sweep's one call site and
// UNREACHABLE cannot fire there; a realckpt-tagged test that cannot even compile is reported as "selected
// (unfiltered) but reported nothing", which reads as an asset or build problem. Open: needs per-cell
// reachability (`go test -tags <cell's Tags> -list '^<test>$' <cell's Pkgs>`). See
// docs/code-notes/cmd-gate.md, "whyNoResult".
func whyNoResult(test string, cells []cell) string {
	for _, c := range cells {
		if c.Run == "" {
			return fmt.Sprintf("selected by cell %q (unfiltered) but reported nothing — "+
				"asset absent, build failure, or the cell died before reaching it", c.Name)
		}
		re, err := regexp.Compile(c.Run)
		if err != nil {
			return fmt.Sprintf("cell %q has an invalid -run %q: %v", c.Name, c.Run, err)
		}
		if re.MatchString(test) {
			return fmt.Sprintf("selected by cell %q (-run %q) but reported nothing — "+
				"asset absent, build failure, or the cell died before reaching it", c.Name, c.Run)
		}
	}
	return "UNREACHABLE: no cell's -run pattern selects this test, so no asset, machine or " +
		"environment could make it run. Fix the pattern or the test name, not the box."
}

// neverConfirmed names a REQUIRED gate that is deliberately absent from the ledger, with the reason. A
// gate here stays permanently FIRST-RUN: its failure is reported as an ITEM and never blocks a tag, so an
// entry is a decision to accept that, not a formality. Remove an entry the moment a real sweep on a box
// with the asset runs the gate; do not let it persist past that.
//
// The map is deliberately a code change with a written reason: `reconcile` is advisory by design, so
// TestParity_everyRequiredGateIsConfirmed (required ⊆ ledger) is the only thing that stops a required
// gate drifting into FIRST-RUN unnoticed, and this map is its only escape hatch.
var neverConfirmed = map[string]string{
	"TestNemotron35LightningReal_oracle": "2026-09-13 — deferred for v0.18.0; needs the ~60GB bf16 checkpoint on the Linux box, not run this release. STILL TRUE 2026-09-18: the asset has still not been pulled to this box (its SKIP was, until today, incorrectly counting as a blocker regardless of this entry — see classifyChecks' neverConfirmed check in the SKIP branch, fixed the same day this was re-confirmed).",

	// TestQwen3NextReal_oracle genuinely runs and cannot fit THIS BOX under the fit-guard's 70% budget; it
	// Skips (not Fatalf) on a decoder.ErrWontFitResident decline, so capacity refusal is not a test failure.
	"TestQwen3NextReal_oracle": "2026-09-18 — needs ~59.3GB (47.3GB int4 weights + 12.0GB KV) against this box's ~43.4GB fit-guard ceiling. It passed on this box at bd085de/2026-08-27; between then and now dfd4bfe9 (2026-09-15, M-28) priced KV from real per-layer geometry (zero for a DeltaNet/mamba/conv mixer layer, the family's actual head_dim otherwise), which moved the estimate up. This one is not a close call either way (59.3GB vs a 43.4GB ceiling, not a few GB over). The gate's own doc comment's \"int4 is ~40GB and fits\" only ever estimated weights and omitted KV entirely. Needs a bigger box.",
}

// awaitingFirstConfirmation names a required gate that has NEVER produced a confirmed result, with the
// date it became required and what will confirm it. It is the third state, and not the same as either
// neighbour:
//
//   - ledger entry: a person looked at a value and said it is correct.
//   - neverConfirmed: we have decided to accept a permanently non-blocking gate.
//   - awaitingFirstConfirmation: nothing has been decided yet, because the gate has not run.
//
// Do not collapse it into either neighbour. Promoting these from "not in the SKIP list, so it must have
// passed" would bank an inferred value as a baseline, which `gate ledger` refuses to do; filing them under
// neverConfirmed would assert a decision to leave them non-blocking forever, the opposite of why they were
// made required. So they are first-run: their failures are ITEMS until a sweep produces a value a person
// promotes. The date is required so an entry that quietly becomes permanent is visible as one.
var awaitingFirstConfirmation = map[string]string{}

// realckptNotRequired names a gate-shaped test in a `//go:build realckpt` file that the sweep RUNS but does
// not require, with the reason. Every such test must be here or in parityRealckptGates;
// TestRealckptGateIsListedOrExplicitlyNotRequired fails otherwise.
//
// An entry does NOT make a failure harmless: the sweep counts an unlisted FAIL as a blocker. What it
// forgoes is a named row in the checkset table, and a SKIP does not block (the asset may not exist on this
// box).
//
// The only acceptable reason is "the family is covered elsewhere", naming the required gate. "This family
// has no required gate anywhere" is the absence of a decision, not a reason: such an entry is a coverage
// hole wearing a reason.
var realckptNotRequired = map[string]string{
	"TestVoxtralReal_gate": "voxtral is required through the tiny gates (decoder TestVoxtral_tinyTextParity and TestVoxtral_tinyCompositionParity); this adds the real Voxtral Mini 3B, whose reference " +
		"needs scripts/pin_voxtral_real.py's output directory (GOINFER_VOXTRAL_REF), not a plain registered asset",
	"TestQwen3ASRReal_gate": "qwen3_asr is required through the tiny gates (decoder TestQwen3ASR_tinyTextParity and TestQwen3ASR_tinyCompositionParity); this adds the real Qwen3-ASR-0.6B, whose reference " +
		"needs scripts/pin_qwen3asr_real.py's output directory (GOINFER_QWEN3ASR_REF), not a plain registered asset",
	"TestGemma3Real_gate": "unregistered asset (GEMMA3_4B, not even GOINFER_-prefixed); gemma3 is " +
		"required through TestGGUF_gemma3_parity + TestForward_logitParity",
	"TestGemma4_26B_gate": "unregistered asset (GOINFER_GEMMA4_26B); gemma4 is required through " +
		"TestGemma4_logitParity + TestGemma4_12B_logitParity",
	"TestLlamaReal_gate": "unregistered asset (GOINFER_QWEN3_REAL); llama is required through " +
		"TestLlama_forwardParity + TestLlama32_forwardParity",
	"TestMistralReal_gate": "unregistered asset (GOINFER_QWEN3_REAL); mistral is required through " +
		"TestMistral_forwardParity",
	"TestNemotron3NanoMoEReal_gate": "nemotron3nano is required through " +
		"TestNemotron3NanoMoE_textParity; this adds the real GGUF checkpoint",
	"TestNemotron3NanoReal_oracle": "nemotron3nano is required through " +
		"TestNemotron3NanoMoE_textParity; this adds the real bf16 checkpoint",
	"TestNemotronReal_gate": "unregistered asset (GOINFER_NEMOTRON_GGUF); nemotron is required " +
		"through TestNemotron_textParity",
	"TestNemotronReal_oracle": "nemotron is required through TestNemotron_textParity; this adds the " +
		"real HF checkpoint",
	"TestQwen2Real_gate": "unregistered asset (GOINFER_QWEN3_REAL); qwen2 is required through " +
		"TestQwen2_forwardParity + TestGGUF_qwen2_parity",
	"TestQwen35Real_gate1SliceParity": "unregistered asset (GOINFER_QWEN35_SLICE_REF); the " +
		"full-model gate TestQwen35Real_gate2FullModel is required",
	"TestQwen3Real_gate": "unregistered asset (GOINFER_QWEN3_REAL); qwen3 is required through " +
		"TestQwen3_forwardParity + TestGGUF_qwen3_parity",
}

// realckptDirs are the packages the realckpt cell runs, and so the packages scanned for its gates.
var realckptDirs = []string{"decoder"}

// legacyRealckptRun is the hand-written -run the realckpt cell used before the pattern was derived. Kept
// ONLY as the fallback for a tree the scan cannot read, and named so a report saying "fell back" is
// unambiguous about which pattern ran.
const legacyRealckptRun = "Qwen35|Real_gate|Real_oracle"

// gateShaped reports whether a test NAME is one this sweep is about. `_gate` and `Parity` are the
// audit's shapes; `_oracle` is here because TestQwen3NextReal_oracle is a REQUIRED gate, so a shape
// rule that excluded it would contradict the list it is checked against.
func gateShaped(name string) bool {
	return strings.HasSuffix(name, "_gate") || strings.HasSuffix(name, "_oracle") ||
		strings.Contains(name, "Parity")
}

var realckptBuildRe = regexp.MustCompile(`(?m)^//go:build (.+)$`)
var realckptWordRe = regexp.MustCompile(`\brealckpt\b`)

// realckptGateTests scans the realckpt-tagged test files for gate-shaped top-level tests.
//
// DERIVED, NOT DECLARED, for the same reason the composition census derives its axes: a
// hand-written -run is a second copy of the gate list, and the two drifted. Only the `//go:build`
// line counts, and only before `package` — decoder/int4_golden_test.go discusses `//go:build
// realckpt` inside a comment while being an ordinary untagged file, and a substring scan pulls it
// into the realckpt cell.
func realckptGateTests(root string) []string { return realckptTests(root, gateShaped) }

// realckptTests is realckptGateTests with the name predicate injected.
func realckptTests(root string, want func(string) bool) []string {
	return buildTaggedTests(root, realckptDirs, realckptWordRe, want)
}

// buildTaggedTests scans dirs for *_test.go files whose //go:build line matches tagWord, and returns the
// sorted set of top-level test function names satisfying want. realckptTests, metalGateTests and
// webgpuGateTests share it, so the scans cannot drift in HOW they read a build line, only in which tag
// they look for.
func buildTaggedTests(root string, dirs []string, tagWord *regexp.Regexp, want func(string) bool) []string {
	seen := map[string]bool{}
	for _, d := range dirs {
		files, _ := filepath.Glob(filepath.Join(root, d, "*_test.go"))
		sort.Strings(files)
		for _, f := range files {
			b, err := os.ReadFile(f)
			if err != nil {
				continue
			}
			src := string(b)
			if i := strings.Index(src, "\npackage "); i >= 0 {
				src = src[:i]
			}
			m := realckptBuildRe.FindStringSubmatch(src)
			if m == nil || !tagWord.MatchString(m[1]) {
				continue
			}
			for _, fm := range funcRe.FindAllStringSubmatch(string(b), -1) {
				if want(fm[1]) {
					seen[fm[1]] = true
				}
			}
		}
	}
	return sortedSet(seen)
}

// metalDirs are the packages scanned for Metal's gate-shaped, goinfer_testhooks-tagged tests, the role
// realckptDirs plays for the decoder-side scan.
var metalDirs = []string{"metal"}

var metalTesthooksWordRe = regexp.MustCompile(`\bgoinfer_testhooks\b`)

// metalGateTests scans metal/'s goinfer_testhooks-tagged test files for gate-shaped top-level tests, the
// Metal analogue of realckptGateTests: a gate that no cell's -run pattern matches would pass `gate gpu` on
// the Mac with nothing saying the cell had nothing to say about it.
func metalGateTests(root string) []string {
	return buildTaggedTests(root, metalDirs, metalTesthooksWordRe, gateShaped)
}

// metalParityRun is metal-parity's -run pattern (metalParity in gpu.go), a named constant so the cell
// definition and TestMetalGateIsListedOrExplicitlyNotRequired read the SAME string rather than two copies
// that can drift.
const metalParityRun = "ResidentParity|residentParity|_bitExact|matchesNonPaged|cpuParity|KernelParity|metalParity|residentIdxParity"

// metalNotRequired names a gate-shaped, goinfer_testhooks-tagged Metal test that
// TestMetalGateIsListedOrExplicitlyNotRequired's scan finds but metalParityRun does not match, with the
// reason it is deliberately excluded. Like realckptNotRequired, an empty reason is an error. Empty: every
// gate-shaped test is covered by metalParityRun; add here, with a reason, if one deliberately is not.
var metalNotRequired = map[string]string{}

// webgpuDirs, webgpuGateTests, webgpuParityRun and webgpuNotRequired are the WebGPU twin of the Metal scan
// above: webgpu-parity must select every gate-shaped goinfer_testhooks-tagged gpu/ test, or that test
// runs in no gate cell and no CI runner.
var webgpuDirs = []string{"gpu"}

// webgpuGateShaped widens gateShaped (capital "Parity", _gate, _oracle) with the forms WebGPU's
// CPU-reference gates use: a lowercase _parity suffix, matchesCPU and matchesSequential.
func webgpuGateShaped(name string) bool {
	return gateShaped(name) || strings.HasSuffix(name, "_parity") || strings.Contains(name, "matchesCPU") ||
		strings.Contains(name, "matchesSequential")
}

// webgpuGateTests scans gpu/'s goinfer_testhooks-tagged test files for gate-shaped top-level tests.
func webgpuGateTests(root string) []string {
	return buildTaggedTests(root, webgpuDirs, metalTesthooksWordRe, webgpuGateShaped)
}

// webgpuParityRun is webgpu-parity's -run pattern (cmd/gate/gpu.go), shared with
// TestWebGPUGateIsListedOrExplicitlyNotRequired so the cell and its check read the same string.
const webgpuParityRun = "Parity|_parity|matchesCPU|matchesSequential"

// webgpuNotRequired names a gate-shaped WebGPU test deliberately left out of webgpuParityRun,
// with the reason. An empty reason is an error, as for metalNotRequired.
var webgpuNotRequired = map[string]string{}

// realckptRun derives the realckpt cell's -run from the tree, and returns a note saying how. Deriving it
// makes "a gate exists" and "the sweep can reach it" the same fact: a hand-widened pattern still left
// required gates matching nothing, and being unlisted as well they were not even reported as DID NOT RUN.
//
// The union with parityRealckptGates is not belt-and-braces: TestQwen35GGUF_weightDiff is required and is
// NOT gate-shaped, so the scan alone would drop it.
func realckptRun() (pattern, note string) {
	root, err := repoRoot()
	if err != nil {
		return legacyRealckptRun, fmt.Sprintf("!! could not locate the repo root (%v) — realckpt "+
			"-run fell back to the hand-written %q, which is known to miss gates", err, legacyRealckptRun)
	}
	set := map[string]bool{}
	for _, t := range realckptGateTests(root) {
		set[t] = true
	}
	scanned := len(set)
	// THE CHANGE IS ADDITIVE ON PURPOSE. legacyRealckptRun's bare "Qwen35" alternative also selects tests that
	// are not gate-shaped (TestQwen35GGUF_vsSafetensors, TestQwen35Real_loaderSlice, ...). Nobody decided they
	// do not belong in a release sweep, and dropping them as a side effect of fixing a FILTER is the silent
	// coverage loss this repo keeps finding, so the derived pattern is unioned with what the old one selects.
	// Removing one is its own change, with its own reason.
	legacy := regexp.MustCompile(legacyRealckptRun)
	carried := realckptTests(root, legacy.MatchString)
	for _, t := range carried {
		set[t] = true
	}
	if scanned == 0 {
		return legacyRealckptRun, fmt.Sprintf("!! no realckpt-tagged gate found under %v — realckpt "+
			"-run fell back to the hand-written %q, which is known to miss gates", realckptDirs, legacyRealckptRun)
	}
	for _, g := range parityRealckptGates {
		set[g.Test] = true
	}
	names := sortedSet(set)
	filterNote := ""
	if runFilter, raw := gateRunFilter(); runFilter != nil {
		var kept []string
		for _, n := range names {
			if runFilter.MatchString(n) {
				kept = append(kept, n)
			}
		}
		filterNote = fmt.Sprintf(", %d after GATE_RUN=%q", len(kept), raw)
		names = kept
	}
	if len(names) == 0 {
		// An empty alternation ("^()$") matches nothing the way an ordinary user would expect, but
		// silently running zero tests and reporting a green cell is exactly the false-confidence
		// shape this whole file exists to catch — so refuse instead of building it.
		return "", fmt.Sprintf("!! GATE_RUN selected zero of %d realckpt gate(s) — nothing would run; "+
			"check the filter", scanned)
	}
	pattern = "^(" + strings.Join(names, "|") + ")$"
	return pattern, fmt.Sprintf("realckpt -run derived from the //go:build realckpt files: %d "+
		"gate-shaped test(s) scanned, %d carried from the legacy pattern, %d selected after the "+
		"union with parityRealckptGates%s", scanned, len(carried), len(sortedSet(set)), filterNote)
}

// unlistedFailures returns the tests that FAILED and are not one of the named gates. The checkset says a
// NAMED gate is green; "nothing else in the sweep failed" is a separate claim, and without it a FAIL in a
// family test outside the checkset leaves the verdict reading ALL REQUIRED GATES GREEN.
//
// EXACT match, unlike catchAllSkips' containment: a test whose name merely CONTAINS a gate name is a
// different test, and hiding its failure is the defect. Exactness also keeps B14 intact: a named gate's
// FIRST-RUN failure matches exactly, so it is excluded here and stays an ITEM.
func unlistedFailures(res *results, checks []gateCheck) []string {
	named := map[string]bool{}
	for _, g := range checks {
		named[g.Test] = true
	}
	seen := map[string]bool{}
	var out []string
	for _, k := range res.topLevel("fail") {
		if named[k.Test] || seen[k.Test] {
			continue
		}
		seen[k.Test] = true
		// Last-writer-wins, the same rule the checkset reads by: a test that failed in the plain
		// cell and passed in the realckpt one is not a failure.
		if act, _ := res.lookupTop(k.Test); act != "fail" {
			continue
		}
		out = append(out, k.Test)
	}
	sort.Strings(out)
	return out
}

// sortedKeys is sortedSet for a string-valued map.
func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// extraBlockers is everything blocking that the CHECKSET CANNOT SEE, and it returns the count so there is
// exactly one place to get the arithmetic wrong: a FAIL in a family parity test outside the checkset, and
// a cell that died without printing a single --- FAIL line. Either would leave the verdict reading ALL
// REQUIRED GATES GREEN.
func extraBlockers(w io.Writer, res *results, checks []gateCheck, cells []cellResult, realckpt bool) int {
	extra := 0

	// THE CATCH-ALL, FOR FAILURES — and here it BLOCKS. Its skip half is advisory because a missing
	// asset on one box is ordinary; a failing test is not, whatever list it is on. No name-shape
	// filter either: the shape filter exists to keep the skip review readable, and a failure nobody
	// wants to read is still a failure.
	if unlisted := unlistedFailures(res, checks); len(unlisted) > 0 {
		fmt.Fprintf(w, "\n-- unlisted FAILURES (blockers): tests that failed and are in no gate list --\n")
		for _, name := range unlisted {
			fmt.Fprintf(w, "   ❌ failed: %s\n", name)
		}
		extra += len(unlisted)
	}

	// A cell that exited non-zero with no --- FAIL line: a panic in a goroutine, a fatal error, a
	// timeout, a build failure. runCell marks it Hidden under RCIsFailure; without counting it the
	// sweep delivers a verdict about a cell that never finished.
	for _, c := range cells {
		if c.Hidden {
			if after, running := res.timeoutPanic(); after != "" {
				fmt.Fprintf(w, "\n   ❌ cell %q TIMED OUT at -timeout %s (rc=%d), not a test failure: in flight %s; every gate after "+
					"it did not run — raise TIMEOUT, or re-run just those gates with GATE_RUN — BLOCKER: %s\n",
					c.Cell.Name, after, c.RC, strings.Join(running, ", "), c.LogPath)
			} else {
				fmt.Fprintf(w, "\n   ❌ cell %q exited rc=%d with zero --- FAIL lines (crash, timeout or "+
					"build failure) — BLOCKER: %s\n", c.Cell.Name, c.RC, c.LogPath)
			}
			extra++
		}
	}

	// The realckpt gates the sweep RUNS but does not require. Printed so "not in the table" is a
	// statement the report makes rather than an absence a reader has to notice.
	if realckpt && len(realckptNotRequired) > 0 {
		fmt.Fprintf(w, "\n-- realckpt gates that run but are NOT required (%d) --\n", len(realckptNotRequired))
		for _, name := range sortedKeys(realckptNotRequired) {
			act, seen := res.lookupTop(name)
			if !seen {
				act = "no result"
			}
			fmt.Fprintf(w, "   %-32s %-9s %s\n", name, act, realckptNotRequired[name])
		}
	}
	return extra
}
