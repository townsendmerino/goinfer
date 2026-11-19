//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/diag/sb2census"
)

// The batched prefill's tails: only ResidualAll reads the [M, hidden]
// residual back to the host; the argmax and all-logits heads read it on the device, and the last-row tails upload one
// row. This gate hashes the output of EVERY tail mode the resident exposes, per fixture, and compares it with the hashes
// recorded from the code before the residual download was dropped (testdata/prefill_tails_baseline.json). It is
// device-specific for the same reason keqv_copy_baseline.json is (the PTX is JIT-compiled to the card's own SASS), so the
// baseline is compared only on the device and driver that recorded it.
//
//	go test -tags 'cuda goinfer_testhooks' -run TestPrefillTails -v ./cuda/
//	GOINFER_TAILS_RECORD=1 go test ...   # (re)writes the baseline: only from code whose tails are the reference
const tailsBaselinePath = "testdata/prefill_tails_baseline.json"

type tailsBaseline struct {
	Device  string                       `json:"device"`
	Driver  string                       `json:"driver"`
	Commit  string                       `json:"recorded_at_commit"`
	Note    string                       `json:"note"`
	Fixture map[string]map[string]string `json:"fixtures"` // fixture -> tail -> sha256 of its output (and the greedy continuation where it has one)
}

// tailsRun returns tail -> hash for one fixture. ok=false: the fixture is not CUDA-resident or has no batched prefill here, so it is not a gate.
func tailsRun(t *testing.T, dir string, moeCache bool) (res map[string]string, ok bool) {
	t.Helper()
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	if moeCache {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
		t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3")
	} else {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "")
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	t.Cleanup(func() { mc.Close() })
	rf, isCUDA := mc.ResidentForwardForTest().(*cudaResident)
	if !isCUDA {
		return nil, false
	}
	if batched, why := rf.PrefillPath(); !batched {
		t.Logf("%s: batched prefill declines (%s)", dir, why)
		return nil, false
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200, 4, 33, 91, 120}
	embs := make([][]float32, len(prompt))
	for i, tok := range prompt {
		embs[i] = mc.EmbedResidentForTest(tok)
	}
	embed := func(id int) []float32 { return mc.EmbedResidentForTest(id) }
	ctx := context.Background()
	res = map[string]string{}
	hex256 := func(h interface{ Sum([]byte) []byte }) string { return hex.EncodeToString(h.Sum(nil)) }

	// last-row logits, then a greedy continuation through decode: the continuation proves the device state the tail leaves behind (r.x, the KV) is the same.
	rf.Reset()
	h := sha256.New()
	logits, err := rf.PrefillLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("%s PrefillLast: %v", dir, err)
	}
	hashLogits(h, logits)
	pos := len(prompt)
	for range 4 {
		tok := keqvArgmax(logits)
		fmt.Fprintf(h, "tok%d", tok)
		if logits, err = rf.Forward(embed(tok), pos); err != nil {
			t.Fatalf("%s decode after PrefillLast: %v", dir, err)
		}
		hashLogits(h, logits)
		pos++
	}
	res["PrefillLast+decode"] = hex256(h)

	rf.Reset()
	rows, err := rf.PrefillLastN(embs, 0)
	if err != nil {
		t.Fatalf("%s PrefillLastN: %v", dir, err)
	}
	h = sha256.New()
	for _, r := range rows {
		hashLogits(h, r)
	}
	res["PrefillLastN"] = hex256(h)

	rf.Reset()
	ids, err := rf.PrefillLastNArgmax(embs, 0)
	if err != nil {
		t.Fatalf("%s PrefillLastNArgmax: %v", dir, err)
	}
	res["PrefillLastNArgmax"] = fmt.Sprint(ids)

	rf.Reset()
	hid, err := rf.HiddenLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("%s HiddenLast: %v", dir, err)
	}
	h = sha256.New()
	hashLogits(h, hid)
	res["HiddenLast"] = hex256(h)

	rf.Reset()
	resid, err := rf.ResidualAll(ctx, embs, 0)
	if err != nil {
		t.Fatalf("%s ResidualAll: %v", dir, err)
	}
	h = sha256.New()
	for _, r := range resid {
		hashLogits(h, r)
	}
	res["ResidualAll"] = hex256(h)
	return res, true
}

func TestPrefillTails_bitIdenticalToTheRecordedBaseline(t *testing.T) {
	// The recorded baselines are of the W4A8 decode, which floatActOffForTest keeps on a Gemma 4 MoE resident; that model's
	// float-activation decode (floatact.go) is held to the CPU by TestGemma4MoE_floatParity instead.
	floatActOffForTest = true
	t.Cleanup(func() { floatActOffForTest = false })
	fixtures := []struct {
		dir      string
		moeCache bool
	}{{"gemma4-dense-scaled", false}, {"gemma4-moe-scaled", false}, {"qwen3moe-tiny-k3", true}}
	dev, drv := gpuIdentity(t)
	record := os.Getenv("GOINFER_TAILS_RECORD") != ""
	var base tailsBaseline
	haveBase := false
	if b, err := os.ReadFile(tailsBaselinePath); err == nil && !record {
		if err := json.Unmarshal(b, &base); err != nil {
			t.Fatalf("parse %s: %v", tailsBaselinePath, err)
		}
		haveBase = true
	}
	out := tailsBaseline{Device: dev, Driver: drv, Fixture: map[string]map[string]string{},
		Note: "Hashes of every batched-prefill tail mode's output (PrefillLast plus a greedy decode continuation, PrefillLastN, PrefillLastNArgmax, HiddenLast, ResidualAll), recorded BEFORE R-24 stopped downloading the full residual on the tails that do not read it. Device-specific: compared only on this GPU and driver. TWO entries were re-recorded after dc338a50 (the expert-major MoE prefill rounds each expert's accumulate once, as the per-row path does): qwen3moe-tiny-k3 PrefillLastN and ResidualAll. The code before that commit reproduces the old hashes with expert-major on and the new ones with GOINFER_CUDA_MOE_EXPERT_MAJOR=0 (the per-row path); the code after it gives the new ones both ways. The other 13 entries are unchanged since 3c1a368b."}
	if c, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output(); err == nil {
		out.Commit = strings.TrimSpace(string(c))
	}
	ran, compared := 0, 0
	for _, fx := range fixtures {
		got, ok := tailsRun(t, "../../testdata/"+fx.dir, fx.moeCache)
		if !ok {
			t.Logf("%s: not CUDA-resident or no batched prefill here, so it is not a gate", fx.dir)
			continue
		}
		ran++
		out.Fixture[fx.dir] = got
		for tail, hsh := range got {
			t.Logf("%s %-20s %.16s", fx.dir, tail, hsh)
		}
		if haveBase && base.Device == dev && base.Driver == drv {
			for tail, hsh := range got {
				want, has := base.Fixture[fx.dir][tail]
				if !has {
					t.Errorf("%s %s: no baseline recorded", fx.dir, tail)
					continue
				}
				compared++
				if hsh != want {
					t.Errorf("%s %s DIFFERS from the pre-change baseline recorded at %s: %s vs %s", fx.dir, tail, base.Commit, hsh, want)
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no fixture ran with a batched prefill: nothing was checked")
	}
	switch {
	case record:
		b, _ := json.MarshalIndent(out, "", " ")
		if err := os.WriteFile(tailsBaselinePath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("recorded %s on %s, driver %s, from %d fixtures", tailsBaselinePath, dev, drv, ran)
	case !haveBase:
		t.Errorf("no %s: record it from the pre-change code with GOINFER_TAILS_RECORD=1", tailsBaselinePath)
	case base.Device != dev || base.Driver != drv:
		t.Logf("baseline is for %s / %s, this is %s / %s: the bit comparison was NOT made", base.Device, base.Driver, dev, drv)
	case compared == 0:
		t.Fatal("the baseline matched this device but nothing was compared")
	default:
		t.Logf("compared %d tail outputs with the baseline recorded at %s: bit-identical", compared, base.Commit)
	}
}

var (
	prefillLaunchUpdate = flag.Bool("prefilllaunch.update", false, "rewrite testdata/cuda_prefill_launch.<goarch>.txt.gz from this tree")
	prefillLaunchDump   = flag.String("prefilllaunch.dump", "", "write every call's whole trace into this directory, one file per golden line")
)

// prefillLaunchExtras are fixtures layer D's census does not build resident (each needs GOINFER_GEMMA4_RESIDENT, which the
// census leaves unset) but prefillCore has branches for: the E-model's per-layer embeddings, Gemma 4's MoE and dense FFNs, two
// layer geometries, and a bidirectional span pass. They are generated, not tracked: scripts/worktree_fixtures.sh copies them
// into a worktree.
var prefillLaunchExtras = []string{"testdata/gemma4-emodel-tiny", "testdata/gemma4-moe-tiny", "testdata/gemma4-dense-twogeom-tiny", "testdata/gemma4-vl-bidir-tiny"}

// TestPrefillLaunch_sequencesHoldTheGolden holds the batched prefill to a recorded launch sequence, the gate for cutting
// prefillCore into phases: on every census fixture and option cell that builds resident, plus prefillLaunchExtras, each of the
// calls prefillLaunchCalls makes issues the launches, copies and allocations the golden records (launchTrace's text, summarised
// per call by traceSummary). A declined call records its error, so the guards are held too. On a red line, run this test with
// -prefilllaunch.dump DIR on this tree and on the commit the golden was recorded at, and diff the two directories.
//
// About 500 loads of tiny fixtures; it streams one heartbeat line per fixture to stderr.
func TestPrefillLaunch_sequencesHoldTheGolden(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "")
	d, err := sb2census.ReadLayerD("../..", "cuda")
	if err != nil {
		t.Fatal(err)
	}
	type load struct {
		row  string
		path string
		o    decoder.Options
	}
	var loads []load
	for _, f := range d.Fixtures {
		for _, c := range sb2census.Cells {
			if !d.Resident[f+" "+c.Name] {
				continue
			}
			path, err := sb2census.Path("../..", f)
			if err != nil {
				t.Fatal(err)
			}
			o := decoder.Options{Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			loads = append(loads, load{f + " " + c.Name, path, o})
		}
	}
	// The census's kv-i8 cell quantizes the CPU cache (KVQuant); the device's int8 cache (kvStore8's quantizing store, the attention's
	// scaled reads) is KVPrecision's, built here on a dense, a MoE, a hybrid, a softcap, a windowed and an image fixture.
	for _, f := range []string{"testdata/llama-tiny", "testdata/qwen3moe-tiny", "testdata/qwen35-tiny", "testdata/gemma2-tiny", "testdata/mistral-tiny-window", "testdata/gemma3-vl-tiny"} {
		loads = append(loads, load{f + " int4/kvprec-i8", filepath.Join("../..", f), decoder.Options{Quant: "int4", KVPrecision: "i8"}})
	}
	for _, f := range prefillLaunchExtras {
		path := filepath.Join("../..", f)
		if _, err := os.Stat(path); err != nil {
			t.Errorf("%s is missing (generated fixtures: scripts/worktree_fixtures.sh): %v", f, err)
			continue
		}
		loads = append(loads, load{f + " int4", path, decoder.Options{Quant: "int4"}})
	}
	if *prefillLaunchDump != "" {
		if err := os.MkdirAll(*prefillLaunchDump, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	var lines []string
	start := time.Now()
	built, steps := 0, 0
	last := ""
	for i, ld := range loads {
		if f, _, _ := strings.Cut(ld.row, " "); f != last {
			last = f
			fmt.Fprintf(os.Stderr, "[prefilllaunch] load %d of %d, %s: %d lines, %s elapsed\n", i+1, len(loads), f, len(lines), time.Since(start).Round(time.Second))
		}
		for _, slots := range []int{0, 3} {
			o := ld.o
			o.Backend, o.ResidentContext, o.ResidentKVSlots = "cuda", 256, slots
			mc, err := decoder.Load(ld.path, o)
			if err != nil {
				if slots == 0 {
					lines = append(lines, ld.row+" load err="+sb2census.Clean(err.Error()))
				}
				continue
			}
			rf, ok := mc.ResidentForwardForTest().(*cudaResident)
			if !ok {
				if slots == 0 {
					lines = append(lines, ld.row+" load not-resident")
				}
				mc.Close()
				continue
			}
			got := prefillLaunchCalls(t, mc, rf, slots > 0)
			if slots == 0 {
				built++
			} else if len(got) > 0 {
				steps++
			}
			for _, c := range got {
				key := ld.row + " " + c.name
				lines = append(lines, key+" "+c.summary)
				if *prefillLaunchDump != "" {
					name := strings.NewReplacer("/", "_", " ", "__").Replace(key) + ".txt"
					if err := os.WriteFile(filepath.Join(*prefillLaunchDump, name), []byte(strings.Join(c.trace, "\n")+"\n"), 0o644); err != nil {
						t.Fatal(err)
					}
				}
			}
			mc.Close()
		}
	}
	t.Logf("%d loads, %d resident, %d with a step; %d lines in %s", len(loads), built, steps, len(lines), time.Since(start).Round(time.Second))
	if built < 40 || steps < 10 {
		t.Errorf("%d resident builds, %d with a step: a gate that barely runs covers little", built, steps)
	}
	golden := sb2census.Golden("testdata", "cuda_prefill_launch")
	if *prefillLaunchUpdate {
		if err := sb2census.Write(golden, lines); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, lines, 3)
	if err != nil {
		t.Fatalf("%v: run with -prefilllaunch.update to record the launch sequences", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the prefill launch sequences differ from %s in %d line(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

type prefillLaunchCall struct {
	name, summary string
	trace         []string
}

// prefillLaunchCalls traces each prefill entry point on rf, each from a reset: the five tails, a prompt continued at a later
// position, chunked passes (prefillChunkCap lowered to 16 rows), the fast levers (the floor knob at 0), image blocks, spans,
// m-RoPE and DeepStack. With step set (a build of several KV slots) it traces only the multi-sequence step and the ragged step,
// with and without a prefill chunk, after untraced prompts in each slot.
func prefillLaunchCalls(t *testing.T, mc *decoder.Model, rf *cudaResident, step bool) []prefillLaunchCall {
	t.Helper()
	_, nLayers, _, _, _, _, vocab := mc.Dims()
	emb := func(id int) []float32 { return mc.EmbedResidentForTest(id % vocab) }
	prompt := func(n, salt int) [][]float32 {
		p := make([][]float32, n)
		for i := range p {
			p[i] = emb(i*37 + salt*11 + 3)
		}
		return p
	}
	p24, p40 := prompt(24, 0), prompt(40, 1)
	ctx := context.Background()
	tr := newLaunchTrace(rf)
	knob := rf.knob
	defer func() { rf.knob = knob; rf.prefillChunkCap.Store(0) }()
	fast := func(name string) (string, bool) {
		if name == "GOINFER_CUDA_FAST_PREFILL_FLOOR" {
			return "0", true
		}
		if knob == nil {
			return os.LookupEnv(name)
		}
		return knob(name)
	}
	var out []prefillLaunchCall
	run := func(name string, chunkCap int, fastOn bool, fn func() error) {
		rf.Reset()
		rf.prefillChunkCap.Store(int64(chunkCap))
		if fastOn {
			rf.knob = fast
		}
		tr.start()
		err := fn()
		lines := tr.stop()
		rf.knob = knob
		rf.prefillChunkCap.Store(0)
		s := traceSummary(lines)
		if err != nil {
			s += " err=" + sb2census.Clean(err.Error())
		}
		out = append(out, prefillLaunchCall{name, s, lines})
	}
	if step {
		if _, hi := rf.BatchStepRange(); hi < 3 {
			return nil
		}
		seed := func() error {
			for s, n := range []int{12, 8, 0} {
				if err := rf.UseKVSlot(s); err != nil {
					return err
				}
				if n > 0 {
					if _, err := rf.PrefillLast(ctx, prompt(n, 10+s), 0); err != nil {
						return err
					}
				}
			}
			return nil
		}
		for _, c := range []struct {
			name string
			seqs func() []decoder.ResidentRaggedSeq
		}{
			{"step", nil},
			{"ragged", func() []decoder.ResidentRaggedSeq {
				return []decoder.ResidentRaggedSeq{{Slot: 0, Pos: 12, Embs: prompt(2, 20)}, {Slot: 1, Pos: 8, Embs: prompt(3, 21), Draw: &decoder.ResidentBatchDraw{Temperature: 0.8, Seed: 7, Draw: 1}}}
			}},
			{"ragged-chunk", func() []decoder.ResidentRaggedSeq {
				return []decoder.ResidentRaggedSeq{{Slot: 2, Pos: 0, Embs: prompt(16, 22), Prefill: &decoder.ResidentRaggedPrefill{PromptLen: 32}}, {Slot: 0, Pos: 12, Embs: prompt(1, 23)}}
			}},
		} {
			rf.Reset()
			if err := seed(); err != nil {
				out = append(out, prefillLaunchCall{c.name, "seed err=" + sb2census.Clean(err.Error()), nil})
				continue
			}
			tr.start()
			var err error
			if c.seqs == nil {
				_, err = rf.StepBatch([]decoder.ResidentBatchSeq{{Slot: 0, Pos: 12, Emb: emb(5)}, {Slot: 1, Pos: 8, Emb: emb(9), Draw: &decoder.ResidentBatchDraw{Temperature: 0.8, Seed: 3, Draw: 2}}})
			} else {
				_, err = rf.StepRagged(c.seqs())
			}
			lines := tr.stop()
			s := traceSummary(lines)
			if err != nil {
				s += " err=" + sb2census.Clean(err.Error())
			}
			out = append(out, prefillLaunchCall{c.name, s, lines})
		}
		rf.Reset()
		return out
	}
	run("last", 0, false, func() error { _, err := rf.PrefillLast(ctx, p24, 0); return err })
	run("lastN", 0, false, func() error { _, err := rf.PrefillLastN(p24, 0); return err })
	run("argmax", 0, false, func() error { _, err := rf.PrefillLastNArgmax(p24, 0); return err })
	run("hidden", 0, false, func() error { _, err := rf.HiddenLast(ctx, p24, 0); return err })
	run("residual", 0, false, func() error { _, err := rf.ResidualAll(ctx, p24, 0); return err })
	run("continue", 0, false, func() error {
		if _, err := rf.PrefillLast(ctx, p24[:12], 0); err != nil {
			return err
		}
		_, err := rf.PrefillLast(ctx, p24[12:], 12)
		return err
	})
	run("chunked", 16, false, func() error { _, err := rf.PrefillLast(ctx, p40, 0); return err })
	run("chunked-hidden", 16, false, func() error { _, err := rf.HiddenLast(ctx, p40, 0); return err })
	run("chunked-residual", 16, false, func() error { _, err := rf.ResidualAll(ctx, p40, 0); return err })
	run("fast", 0, true, func() error { _, err := rf.PrefillLast(ctx, p40, 0); return err })
	run("fast-chunked", 16, true, func() error { _, err := rf.PrefillLast(ctx, p40, 0); return err })
	run("image", 0, false, func() error {
		_, err := rf.PrefillImageBlocksLast(ctx, p24, 0, [][2]int{{4, 10}, {14, 18}})
		return err
	})
	run("span", 0, false, func() error { _, err := rf.PrefillSpansLast(ctx, p24, 0, [][2]int{{4, 10}}); return err })
	mpos := make([][3]int, len(p24))
	for i := range mpos {
		mpos[i] = [3]int{i, i, i}
		if i >= 4 && i < 10 {
			mpos[i] = [3]int{4, (i - 4) / 3, (i - 4) % 3}
		}
	}
	run("mrope", 0, false, func() error { _, err := rf.PrefillMRoPELast(ctx, p24, 0, mpos); return err })
	deep := make([][]float32, min(nLayers, 2))
	for l := range deep {
		deep[l] = make([]float32, 6*rf.hidden)
		for i := range deep[l] {
			deep[l][i] = float32((i+l)%7) * 0.01
		}
	}
	run("deepstack", 0, false, func() error { _, err := rf.PrefillMRoPEDeepstackLast(ctx, p24, 0, mpos, deep, 4, 6); return err })
	rf.Reset()
	return out
}

// launchTrace records the device work one prefill call issues, as text a refactor can be held to: every kernel launch (its
// pipeline's field name on cudaResident, its geometry, every argument), every copy and every allocation and free, in issue order.
//
// A device address is not stable from run to run, so an argument names it by what it points into: "aN+off" for byte off of the
// N-th allocation made during the trace, "pN+off" for an address in the N-th buffer that existed before the trace started (one
// seen first in that order: such a buffer's extent is unknown, so every distinct address of one gets its own N). An 8-byte argument
// is read as an address when it is at least 1<<32 and as a scalar otherwise; every scalar is printed as its bits.
type launchTrace struct {
	r     *cudaResident
	names map[uintptr]string // pipeline -> field name, cached across traces

	mu     sync.Mutex
	lines  []string
	allocs []traceAlloc // allocations made during the trace, newest last
	pre    map[uint64]int
	unk    map[uintptr]int

	restore func()
}

type traceAlloc struct {
	base uint64
	n    int
	live bool
}

func newLaunchTrace(r *cudaResident) *launchTrace {
	t := &launchTrace{r: r, names: map[uintptr]string{}}
	v := reflect.ValueOf(r).Elem()
	pt := reflect.TypeFor[Pipeline]()
	var walk func(name string, f reflect.Value)
	walk = func(name string, f reflect.Value) {
		switch {
		case f.Type() == pt:
			p := reflect.NewAt(pt, unsafe.Pointer(f.UnsafeAddr())).Elem().Interface().(Pipeline)
			if k := p.Ptr(); k != 0 {
				if _, dup := t.names[k]; !dup {
					t.names[k] = name
				}
			}
		case f.Kind() == reflect.Array:
			for i := range f.Len() {
				walk(fmt.Sprintf("%s[%d]", name, i), f.Index(i))
			}
		}
	}
	for i := range v.NumField() {
		walk(v.Type().Field(i).Name, v.Field(i))
	}
	return t
}

// start begins a trace and installs the hooks; stop removes them and returns the recorded lines.
func (t *launchTrace) start() {
	t.lines, t.allocs, t.pre, t.unk = nil, nil, map[uint64]int{}, map[uintptr]int{}
	gpu.TraceForTest = t.event
	prev := prefillScratchSeenForTest
	prefillScratchSeenForTest = func(bytes int64) {
		t.add(fmt.Sprintf("scratch %d", bytes))
		if prev != nil {
			prev(bytes)
		}
	}
	t.restore = func() { gpu.TraceForTest, prefillScratchSeenForTest = nil, prev }
}

func (t *launchTrace) stop() []string {
	t.restore()
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.lines
}

func (t *launchTrace) add(s string) {
	t.mu.Lock()
	t.lines = append(t.lines, s)
	t.mu.Unlock()
}

// addr names a device address (see launchTrace). Called with t.mu held.
func (t *launchTrace) addr(a uint64) string {
	if a == 0 {
		return "null"
	}
	for i := len(t.allocs) - 1; i >= 0; i-- {
		if x := t.allocs[i]; x.live && a >= x.base && a < x.base+uint64(max(x.n, 1)) {
			return fmt.Sprintf("a%d+%d", i, a-x.base)
		}
	}
	n, ok := t.pre[a]
	if !ok {
		n = len(t.pre)
		t.pre[a] = n
	}
	return fmt.Sprintf("p%d", n)
}

func (t *launchTrace) pipe(p Pipeline) string {
	k := p.Ptr()
	if n, ok := t.names[k]; ok {
		return n
	}
	n, ok := t.unk[k]
	if !ok {
		n = len(t.unk)
		t.unk[k] = n
	}
	return fmt.Sprintf("?%d", n)
}

func (t *launchTrace) event(ev gpu.TraceEvent) {
	t.mu.Lock()
	defer t.mu.Unlock()
	var s string
	switch ev.Op {
	case gpu.TraceLaunch:
		var b strings.Builder
		c := ev.Cfg
		fmt.Fprintf(&b, "launch %s g=%d,%d,%d b=%d,%d,%d sh=%d", t.pipe(ev.P), c.GridX, c.GridY, c.GridZ, c.BlockX, c.BlockY, c.BlockZ, c.SharedMemBytes)
		for _, a := range ev.Args {
			bits, n := a.Raw()
			switch {
			case n == 8 && (bits >= 1<<32 || bits == 0):
				b.WriteString(" " + t.addr(bits))
			default:
				fmt.Fprintf(&b, " %d:%x", n, bits)
			}
		}
		s = b.String()
	case gpu.TraceUpload:
		h := sha256.Sum256(ev.Src)
		s = fmt.Sprintf("upload %s %d %s", t.addr(ev.Dst), ev.Bytes, hex.EncodeToString(h[:6]))
	case gpu.TraceDownload:
		s = fmt.Sprintf("download %s %d", t.addr(ev.Dst), ev.Bytes)
	case gpu.TraceUploadAsync:
		s = fmt.Sprintf("upload-async %s %d", t.addr(ev.Dst), ev.Bytes)
	case gpu.TraceZeroAsync:
		s = fmt.Sprintf("zero-async %s %d", t.addr(ev.Dst), ev.Bytes)
	case gpu.TraceCopyDevice:
		s = fmt.Sprintf("copy %s <- %s %d", t.addr(ev.Dst), t.addr(ev.SrcPtr), ev.Bytes)
	case gpu.TraceAlloc:
		t.allocs = append(t.allocs, traceAlloc{base: ev.Dst, n: ev.Bytes, live: true})
		s = fmt.Sprintf("alloc a%d %d %s", len(t.allocs)-1, ev.Bytes, ev.Line)
	case gpu.TraceFree:
		s = "free " + t.addr(ev.Dst)
		for i := len(t.allocs) - 1; i >= 0; i-- {
			if t.allocs[i].live && t.allocs[i].base == ev.Dst {
				t.allocs[i].live = false
				break
			}
		}
	default:
		s = fmt.Sprintf("op%d", ev.Op)
	}
	t.lines = append(t.lines, s)
}

// traceSummary is one golden line's payload for a trace: the count of each kind of line and a hash of the whole sequence.
func traceSummary(lines []string) string {
	counts := map[string]int{}
	h := sha256.New()
	for _, l := range lines {
		k, _, _ := strings.Cut(l, " ")
		counts[k]++
		h.Write([]byte(l))
		h.Write([]byte{'\n'})
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s=%d ", k, counts[k])
	}
	fmt.Fprintf(&b, "sha=%s", hex.EncodeToString(h.Sum(nil))[:16])
	return b.String()
}
