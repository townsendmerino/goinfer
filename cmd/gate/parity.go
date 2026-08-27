package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
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
	{"qwen3-gguf", "TestGGUF_qwen3_parity"},
	{"qwen2moe", "TestQwen2Moe_forwardParity"},
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
	// The int4 FORWARD gate (23 fixtures / 16 architectures) is the broadest quant check here, and
	// it was missing from this list until 2026-08-26 -- so when it went red in the v0.15.0-prep
	// sweep it surfaced only as an anonymous "1 fail" with no gate name, and attributing it took
	// hours. A required gate that is not named here is a gate whose failure nobody can read.
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
}

// emitGates are the numeric-oracle gates expected to record a manifest row under EMIT_MANIFEST.
// Family here is the manifest family the gate writes, which is why the pair is the other way round
// from the lists above — a detail that once produced six rows with the columns swapped.
var emitGates = []gateCheck{
	{"phi3", "TestPhi3MiniReal_gate"},
	{"deepseek_v2", "TestDeepseekV2LiteReal_gate"},
	{"deepseek_v3", "TestDeepseekMoonlightReal_gate"},
	{"qwen3_5_moe", "TestQwen35Real_gate2FullModel"},
	{"cohere", "TestCohereAyaReal_gate"},
	{"cohere2", "TestCohere2R7bReal_gate"},
	{"qwen3_next", "TestQwen3NextReal_oracle"},
}

// assetNeverBuilt names required gates whose asset has NEVER been built anywhere, so no invocation
// can make them green. They are reported and counted as coverage gaps, not blockers.
//
// THE LIST IS EMPTY, AND IT GOT THERE THE ONLY CORRECT WAY (2026-08-18, v1.0 gate 1.3):
// TestW4A8DecodeParity was the sole entry — it needs a MATCHED int4+int8 .giw pair, and only int4
// bundles had ever been produced — until the pair was built from one source GGUF (so "matched" is by
// construction, not by belief) and the gate ran green on first invocation. The machinery stays: the
// next gate whose asset has never been built belongs here, and an empty list is the honest current
// state rather than a reason to delete the classification. The only correct way OFF this list is to
// build the asset.
var assetNeverBuilt = map[string]bool{}

// parityCells builds the sweep's two cells. env carries whatever the asset preflight resolved.
func parityCells(env map[string]string, realckpt bool, timeout string) []cell {
	base := cell{
		Name:    "./decoder/ ./tokenizer/",
		Pkgs:    []string{"./decoder/", "./tokenizer/"},
		Timeout: timeout,
		Env:     env,
	}
	cells := []cell{base}
	if realckpt {
		cells = append(cells, cell{
			Name:    "realckpt real-model gates",
			Pkgs:    []string{"./decoder/"},
			Tags:    []string{"realckpt"},
			Run:     "Qwen35|Real_gate",
			Timeout: timeout,
			Env:     env,
		})
	}
	return cells
}

// assetPreflight resolves the asset environment from the SHARED REGISTRY (testdata/assets.json) and
// reports what it resolved.
//
// This exists because the same invocation error produced a false "15 BLOCKER(S)" three separate
// times while the tree was fine every time: the gates skip-if-absent and a skip is reported as a
// blocker, so an unset variable and a genuinely missing checkpoint were indistinguishable in the
// output. The registry is the single implementation of "is this asset present" — an earlier table
// inside the sweep tested `[ -e "$path" ]`, which a DIRECTORY satisfies, so it reported resolved for
// three entries where the loader wanted the .gguf FILE inside.
//
// A preflight that does not run is announced LOUDLY rather than left to become a blocker cascade:
// without python3 every asset-gated gate skips and the count is about the failure, not the tree.
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
	for k, v := range parseExports(path) {
		env[k] = v
	}
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
	out, err := exec.Command("python3", "scripts/gate_ledger.py", "classify", "--gate", name).Output()
	if err != nil {
		return "CONFIRMED"
	}
	if c := strings.TrimSpace(string(out)); c != "" {
		return c
	}
	return "CONFIRMED"
}

// runParity is the sweep's decision. Returns the process exit code.
func runParity(w io.Writer, logDir string) int {
	realckpt := env("REALCKPT", "1") == "1"
	emitManifest := env("EMIT_MANIFEST", "0") == "1"
	timeout := env("TIMEOUT", "120m")

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

	// THE COMPOSITION, NOT JUST THE VERDICT. This gate's axes are family × quant × loader, and a
	// pass COUNT alone cannot distinguish "the axes are covered" from "an axis collapsed to one
	// value" — which is exactly how the forward goldens stayed f32-only through nine refreshes
	// behind an accurate count.
	composition(w, false)

	cfg := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure: false}
	cfg.Cells = parityCells(cellEnv, realckpt, timeout)
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

	rows, blockers, gaps, firstRuns := classifyChecks(res, checks, ledgerClassify)
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

	if emitManifest {
		blockers += mergeManifest(w, res, logDir)
	}

	fmt.Fprintln(w)
	if firstRuns > 0 {
		fmt.Fprintf(w, "== %d FIRST-RUN: failed with no confirmed prior result ==\n", firstRuns)
		fmt.Fprintf(w, "   Reported, NOT counted as blockers -- there is no second point to compute a delta from.\n")
		fmt.Fprintf(w, "   NOT a claim they are harmless: each is an ITEM. Confirm a value deliberately with\n")
		fmt.Fprintf(w, "   scripts/gate_ledger.py promote --gate <G> --value <V> --by <you>.\n")
	}
	rec := exec.Command("python3", "scripts/gate_ledger.py", "reconcile", "--gates", strings.Join(checked, ","))
	rec.Stdout, rec.Stderr = w, w
	_ = rec.Run() // advisory, exactly as the shell had it
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
func classifyChecks(res *results, checks []gateCheck, ledger func(string) string) (rows []checkRow, blockers, gaps, firstRuns int) {
	for _, g := range checks {
		act, seen := res.lookupTop(g.Test)
		var mark string
		switch {
		case !seen:
			// THE OUTCOME A TALLY CANNOT SEE. A required gate that produced no result at all — a
			// renamed test, a -run filter that stopped matching, a package that failed to build —
			// leaves a pass/skip/fail count entirely undisturbed. It is a blocker.
			mark = "⛔ DID NOT RUN (blocker)"
			blockers++
		case act == "pass":
			mark = "✅ pass"
		case act == "fail":
			// THE FOURTH OUTCOME (B14). A gate failing with no confirmed prior result is asserting a
			// DELTA IT HAS NO SECOND POINT TO COMPUTE, so it is an ITEM, not a blocker — the change
			// that made a failure visible is the one change that provably did not cause it. This is
			// about ATTRIBUTION, not harmlessness: the first observed value must not be banked as a
			// baseline without a person deciding it is correct (gate_ledger.py promote).
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
			if assetNeverBuilt[g.Test] {
				mark = "COVERAGE GAP - asset never built (NOT a blocker; see assetNeverBuilt)"
				gaps++
			} else {
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
		switch {
		case fams[g.Family]:
			fmt.Fprintf(w, "   %-34s ✅ recorded row (%s)\n", g.Test, g.Family)
		case seen && act == "pass":
			fmt.Fprintf(w, "   %-34s ⚠️  PASSED but emitted NO row for %s (emitter missing?)\n", g.Test, g.Family)
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
	j := strings.Index(rest, `"`)
	if j < 0 {
		return ""
	}
	return rest[:j]
}

func sortedSet(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
