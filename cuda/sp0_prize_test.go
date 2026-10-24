//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSP0_prize is SP0's instrument (docs/measurements/sp0-2026-10/README.md, which registers the protocol): a drafter's
// acceptance against the target's own greedy continuation by teacher forcing, which needs no rollback in the target, and the
// target's verify-row cost per call. It is a measurement tool, not a check: it skips unless SP0_TARGET is set.
//
//	GOINFER_HEAVY_TESTS=1 SP0_TARGET=<model> SP0_KIND=dflash|ngram|validate [SP0_DRAFTER=<dflash dir>] SP0_OUT=<out.jsonl> \
//	  [SP0_SUITES=code,math] [SP0_THINK=template|on|off] [SP0_MAXNEW=256] [SP0_COST_CTX=512,4096] [SP0_COST_REPS=20] [SP0_SKIP_COST=1] [SP0_MOE_CACHE=1] [SP0_CTX=4608] [SP0_LOADONLY=1] [SP0_SKIP_ACCEPT=1] \
//	  go test -tags 'cuda goinfer_testhooks' -run '^TestSP0_prize$' -v -timeout 90m ./cuda/
//
// validate runs the real block-spec loop (BlockSpec.Generate) beside the simulation on a dense pairing and records both, so
// the simulation is checked against the loop it replays. Progress goes to stderr as it happens.
func TestSP0_prize(t *testing.T) {
	target := os.Getenv("SP0_TARGET")
	if target == "" {
		t.Skip("set SP0_TARGET, SP0_KIND and SP0_OUT")
	}
	requireHeavyModel(t)
	kind := os.Getenv("SP0_KIND")
	if kind != "dflash" && kind != "ngram" && kind != "validate" {
		t.Fatalf("SP0_KIND %q: want dflash, ngram or validate", kind)
	}
	out, err := os.Create(os.Getenv("SP0_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	rec := func(v map[string]any) {
		if err := enc.Encode(v); err != nil {
			t.Fatal(err)
		}
	}
	start := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "sp0 [%6.1fs] %s\n", time.Since(start).Seconds(), fmt.Sprintf(format, a...))
	}

	var dr *decoder.DFlashDrafter
	// f32 KV: the CUDA drafter attends with this resident's pipelines and refuses an f16 cache (cudaResident.AttachDrafter).
	opts := decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: sp0EnvInt(t, "SP0_CTX", 4608), KVPrecision: "f32", MoECacheExperts: os.Getenv("SP0_MOE_CACHE") == "1"}
	if kind != "ngram" {
		dr, err = decoder.LoadDFlashDrafter(os.Getenv("SP0_DRAFTER"))
		if err != nil {
			t.Fatalf("load drafter: %v", err)
		}
		defer dr.Close()
		opts = withDrafterReserve(opts, dr)
	}
	t0 := time.Now()
	m, err := decoder.Load(target, opts)
	if err != nil {
		t.Fatalf("load %s: %v", target, err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s is not CUDA-resident: %s", target, m.ResidentDecline())
	}
	tk, err := modelload.Tokenizer(target)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	pb, pr := m.PrefillPath()
	vb, vr := r.VerifyPath()
	meta := map[string]any{"kind": "meta", "target": target, "sp0_kind": kind, "load_s": time.Since(t0).Seconds(),
		"prefill_batched": pb, "prefill_reason": pr, "verify_batched": vb, "verify_reason": vr, "ctx_cap": m.ResidentContextCap()}
	if dr != nil {
		meta["drafter"], meta["block"], meta["taps"] = os.Getenv("SP0_DRAFTER"), dr.BlockSize(), dr.TargetLayerIDs()
	}
	rec(meta)
	logf("loaded %s in %.1fs (prefill batched %v: %s; verify batched %v: %s)", target, time.Since(t0).Seconds(), pb, pr, vb, vr)
	if os.Getenv("SP0_LOADONLY") == "1" {
		if dr != nil {
			if _, e := r.AttachBlockDrafter(dr); e != nil {
				t.Fatalf("AttachBlockDrafter: %v", e)
			}
		}
		logf("load only: the load and the drafter attach fit")
		return // the job's fit check: the same load, then stop
	}

	maxNew := sp0EnvInt(t, "SP0_MAXNEW", 256)
	think := os.Getenv("SP0_THINK")
	suites := strings.Split(sp0EnvStr("SP0_SUITES", "code,math"), ",")
	type item struct {
		suite  string
		idx    int
		prompt []int
		cont   []int
	}
	var items []item
	for _, s := range suites {
		texts, ok := sp0Suites()[s]
		if !ok {
			t.Fatalf("SP0_SUITES: no suite %q", s)
		}
		for i, text := range texts {
			ids, e := sp0Encode(tk, text, think)
			if e != nil {
				t.Fatalf("encode %s/%d: %v", s, i, e)
			}
			items = append(items, item{suite: s, idx: i, prompt: ids})
		}
	}

	// Phase 1: every teacher continuation, before the resident is driven directly (its reuse bookkeeping is the model's).
	var spec *decoder.BlockSpec
	if kind == "validate" {
		spec, err = m.NewBlockSpec(dr, dr.TargetLayerIDs())
		if err != nil {
			t.Fatalf("NewBlockSpec: %v", err)
		}
	}
	for i := range items {
		it := &items[i]
		tg := time.Now()
		ch, g := m.Generate(context.Background(), it.prompt, maxNew, decoder.SamplingParams{})
		for id := range ch {
			it.cont = append(it.cont, id)
		}
		if e := g.Err(); e != nil {
			t.Fatalf("generate %s/%d: %v", it.suite, it.idx, e)
		}
		line := map[string]any{"kind": "teacher", "suite": it.suite, "idx": it.idx, "prompt_len": len(it.prompt), "cont_len": len(it.cont),
			"tok_s": float64(len(it.cont)) / time.Since(tg).Seconds()}
		if kind == "validate" {
			var commits []int
			got, rounds, e := spec.Generate(it.prompt, decoder.BlockSpecOptions{MaxTokens: maxNew, VerifyWidth: 8,
				OnRound: func(_, c int) { commits = append(commits, c) }})
			if e != nil {
				t.Fatalf("BlockSpec.Generate %s/%d: %v", it.suite, it.idx, e)
			}
			line["loop_rounds"], line["loop_commits"], line["loop_equals_greedy"] = rounds, commits, sp0Equal(got, it.cont)
		}
		rec(line)
		logf("teacher %d/%d %s/%d: %d prompt + %d generated ids", i+1, len(items), it.suite, it.idx, len(it.prompt), len(it.cont))
	}
	spec = nil
	var fill []int // the cost arms' context: the prompts and continuations, so MoE routing sees real text
	for _, it := range items {
		fill = append(append(fill, it.prompt...), it.cont...)
	}

	// Phase 2: acceptance by teacher forcing.
	if kind == "ngram" {
		for i, it := range items {
			for _, k := range []int{1, 2, 3, 4, 6, 8} {
				hist, rounds := sp0Ngram(it.prompt, it.cont, k)
				rec(map[string]any{"kind": "accept_ngram", "suite": it.suite, "idx": it.idx, "k": k, "rounds": rounds,
					"tokens": len(it.cont) - 1, "drafted_hist": hist})
			}
			logf("ngram %d/%d %s/%d done", i+1, len(items), it.suite, it.idx)
		}
	} else {
		rd, e := r.AttachBlockDrafter(dr)
		if e != nil {
			t.Fatalf("AttachBlockDrafter: %v", e)
		}
		taps := dr.TargetLayerIDs()
		widths := sp0Widths(dr.BlockSize())
		if kind == "validate" {
			widths = []int{8}
		}
		for i, it := range items {
			if os.Getenv("SP0_SKIP_ACCEPT") == "1" {
				break // a cost-only re-run: the teacher continuations above are the cost arms' fill text
			}
			seq := append(append([]int(nil), it.prompt...), it.cont...)
			capt, e := sp0Capture(m, r, taps, seq)
			if e != nil {
				t.Fatalf("capture %s/%d: %v", it.suite, it.idx, e)
			}
			for _, w := range widths {
				commits, e := sp0SimulateDFlash(m, rd, dr, capt, len(it.prompt), it.cont, w)
				if e != nil {
					t.Fatalf("simulate %s/%d w=%d: %v", it.suite, it.idx, w, e)
				}
				rec(map[string]any{"kind": "accept_dflash", "suite": it.suite, "idx": it.idx, "w": w, "rounds": len(commits),
					"tokens": len(it.cont) - 1, "commits": commits})
			}
			logf("dflash %d/%d %s/%d: %d widths simulated", i+1, len(items), it.suite, it.idx, len(widths))
		}
		if kind != "validate" && os.Getenv("SP0_SKIP_COST") == "" {
			sp0Cost(t, m, r, rd, dr, fill, widths, rec, logf)
		}
		return
	}
	if os.Getenv("SP0_SKIP_COST") == "" {
		sp0Cost(t, m, r, nil, nil, fill, []int{2, 3, 4, 5, 6, 7, 8, 9}, rec, logf) // a round drafts 0 to 8 tokens
	}
}

// sp0Widths is the registered DFlash ladder, clamped to the drafter's block.
func sp0Widths(block int) []int {
	var ws []int
	for _, w := range []int{2, 3, 4, 6, 8, 12, 16} {
		if w <= block {
			ws = append(ws, w)
		}
	}
	return ws
}

// sp0Capture prefills seq from position 0 with the taps armed and returns [tap][len(seq)*hidden].
func sp0Capture(m *decoder.Model, r *cudaResident, taps []int, seq []int) ([][]float32, error) {
	r.Reset()
	if err := r.SetBatchedCapture(taps); err != nil {
		return nil, err
	}
	defer func() { _ = r.SetBatchedCapture(nil) }()
	if _, err := r.PrefillSeedArgmax(sp0Embed(m, seq), 0); err != nil {
		return nil, err
	}
	capt := r.BatchedCapture()
	hidden := len(m.EmbedResidentForTest(seq[0]))
	if len(capt) != len(taps) {
		return nil, fmt.Errorf("capture has %d taps, want %d", len(capt), len(taps))
	}
	out := make([][]float32, len(capt))
	for i, c := range capt {
		if len(c) != len(seq)*hidden {
			return nil, fmt.Errorf("tap %d captured %d floats, want %d rows x %d", taps[i], len(c), len(seq), hidden)
		}
		out[i] = append([]float32(nil), c...)
	}
	return out, nil
}

// sp0SimulateDFlash replays BlockSpec.generate's rounds at width w against the known continuation: the drafter's context holds the
// prompt's rows, each round drafts [anchor, mask x (w-1)], accepts the prefix that matches cont, and folds the anchor's and the
// accepted drafts' rows, as generate folds a verify pass's capture. It returns each round's committed count (1 + accepted).
func sp0SimulateDFlash(m *decoder.Model, rd decoder.ResidentBlockDrafter, dr *decoder.DFlashDrafter, capt [][]float32, promptLen int, cont []int, w int) ([]int, error) {
	hidden := len(capt[0]) / (promptLen + len(cont))
	fuse := func(lo, n int) error {
		cat := make([][]float32, n)
		for i := range n {
			row := make([]float32, 0, len(capt)*hidden)
			for _, tp := range capt {
				row = append(row, tp[(lo+i)*hidden:(lo+i+1)*hidden]...)
			}
			cat[i] = row
		}
		fused, err := rd.FuseContext(cat)
		if err != nil {
			return err
		}
		return rd.ExtendContext(fused)
	}
	rd.TruncateContext(0)
	if err := fuse(0, promptLen); err != nil {
		return nil, err
	}
	mask := m.EmbedResidentForTest(dr.MaskTokenID())
	var commits []int
	for j := 0; j+1 < len(cont); {
		blockIn := make([][]float32, w)
		blockIn[0] = m.EmbedResidentForTest(cont[j])
		for i := 1; i < w; i++ {
			blockIn[i] = mask
		}
		trunk, err := rd.DraftBlock(blockIn)
		if err != nil {
			return nil, err
		}
		drafted, err := rd.DraftTokens(trunk[1:])
		if err != nil {
			return nil, err
		}
		avail := len(cont) - 1 - j
		accepted := 0
		for i := 0; i < len(drafted) && i < avail && drafted[i] == cont[j+1+i]; i++ {
			accepted = i + 1
		}
		if accepted == avail {
			commits = append(commits, accepted) // the continuation ends inside this round: no target token after it
			break
		}
		commits = append(commits, 1+accepted)
		if err := fuse(promptLen+j, 1+accepted); err != nil {
			return nil, err
		}
		j += 1 + accepted
	}
	return commits, nil
}

// sp0Ngram replays the n-gram loop at k drafted tokens: each round drafts from the prompt and the continuation so far, accepts the
// matching prefix and commits it plus the target's own token. It returns how many rounds drafted each count (0 is a plain step).
func sp0Ngram(prompt, cont []int, k int) (map[string]int, int) {
	d := &decoder.NgramDrafter{}
	hist := map[string]int{}
	rounds := 0
	for j := 0; j+1 < len(cont); {
		ctx := append(append([]int(nil), prompt...), cont[:j+1]...)
		avail := len(cont) - 1 - j
		draft := d.Draft(ctx, min(k, avail))
		accepted := 0
		for i := 0; i < len(draft) && draft[i] == cont[j+1+i]; i++ {
			accepted = i + 1
		}
		hist[strconv.Itoa(len(draft))]++
		rounds++
		j += min(1+accepted, avail)
	}
	return hist, rounds
}

// sp0Cost times the registered arms at each context length: the plain step, PrefillLastNArgmax and ForwardN at each width, and the
// drafter's block when one is attached, rotated per repetition and reversed on alternate ones.
func sp0Cost(t *testing.T, m *decoder.Model, r *cudaResident, rd decoder.ResidentBlockDrafter, dr *decoder.DFlashDrafter, fill []int,
	widths []int, rec func(map[string]any), logf func(string, ...any)) {
	t.Helper()
	reps := sp0EnvInt(t, "SP0_COST_REPS", 20)
	const warm = 3
	for _, cs := range strings.Split(sp0EnvStr("SP0_COST_CTX", "512,4096"), ",") {
		L, err := strconv.Atoi(cs)
		if err != nil {
			t.Fatalf("SP0_COST_CTX %q: %v", cs, err)
		}
		maxW := widths[len(widths)-1]
		for len(fill) < L+maxW+1 {
			fill = append(fill, fill...)
		}
		r.Reset()
		if dr != nil {
			if err := r.SetBatchedCapture(dr.TargetLayerIDs()); err != nil {
				t.Fatal(err)
			}
		}
		// In pieces of one pass: an armed capture keeps a batched prefill from chunking itself, and one M=4096 pass does not fit beside
		// the drafter on an 8 GB card. Each piece's capture rows are joined in order.
		const piece = 512
		var capt [][]float32
		for lo := 0; lo < L; lo += piece {
			hi := min(lo+piece, L)
			if _, err := r.PrefillSeedArgmax(sp0Embed(m, fill[lo:hi]), lo); err != nil {
				t.Fatalf("cost fill [%d,%d) of %d: %v", lo, hi, L, err)
			}
			if dr != nil {
				c := r.BatchedCapture()
				if capt == nil {
					capt = make([][]float32, len(c))
				}
				for i := range c {
					capt[i] = append(capt[i], c[i]...)
				}
			}
		}
		if dr != nil {
			_ = r.SetBatchedCapture(nil)
			hidden := len(capt[0]) / L
			cat := make([][]float32, L)
			for i := range L {
				row := make([]float32, 0, len(capt)*hidden)
				for _, tp := range capt {
					row = append(row, tp[i*hidden:(i+1)*hidden]...)
				}
				cat[i] = row
			}
			rd.TruncateContext(0)
			for lo := 0; lo < L; lo += piece {
				fused, err := rd.FuseContext(cat[lo:min(lo+piece, L)])
				if err == nil {
					err = rd.ExtendContext(fused)
				}
				if err != nil {
					t.Fatalf("drafter context to %d: %v", L, err)
				}
			}
		}
		rows := sp0Embed(m, fill[L:L+maxW])
		mask := m.EmbedResidentForTest(0)
		if dr != nil {
			mask = m.EmbedResidentForTest(dr.MaskTokenID())
		}
		type arm struct {
			name string
			w    int
			run  func() error
		}
		arms := []arm{{"step", 1, func() error { _, e := r.ForwardArgmax(rows[0], L); return e }}}
		for _, w := range widths {
			arms = append(arms,
				arm{"verifyN", w, func() error { _, e := r.PrefillLastNArgmax(rows[:w], L); return e }},
				arm{"forwardN", w, func() error { _, e := r.ForwardN(rows[:w], L); return e }})
			if rd != nil {
				arms = append(arms, arm{"draft", w, func() error {
					in := make([][]float32, w)
					in[0] = rows[0]
					for i := 1; i < w; i++ {
						in[i] = mask
					}
					trunk, e := rd.DraftBlock(in)
					if e == nil {
						_, e = rd.DraftTokens(trunk[1:])
					}
					return e
				}})
			}
		}
		ms := make([][]float64, len(arms))
		for rep := range warm + reps {
			order := make([]int, len(arms))
			for i := range order {
				order[i] = (i + rep) % len(arms)
			}
			if rep%2 == 1 {
				sort.Sort(sort.Reverse(sort.IntSlice(order)))
			}
			for _, ai := range order {
				t0 := time.Now()
				if err := arms[ai].run(); err != nil {
					t.Fatalf("cost %s w=%d at %d: %v", arms[ai].name, arms[ai].w, L, err)
				}
				if rep >= warm {
					ms[ai] = append(ms[ai], float64(time.Since(t0).Microseconds())/1000)
				}
			}
			logf("cost ctx %d: repetition %d/%d", L, rep+1, warm+reps)
		}
		for i, a := range arms {
			rec(map[string]any{"kind": "cost", "ctx": L, "arm": a.name, "w": a.w, "ms": ms[i]})
		}
	}
}

func sp0Embed(m *decoder.Model, ids []int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = m.EmbedResidentForTest(id)
	}
	return out
}

// sp0Encode renders one user message through the checkpoint's template with the given thinking mode ("" or template: the
// template's default) and encodes it as the chat routes do.
func sp0Encode(tk *tokenizer.Tokenizer, text, think string) ([]int, error) {
	tm, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		return nil, err
	}
	switch think {
	case "", "template":
	case "on":
		tm = tm.WithThinking(chat.ThinkOn)
	case "off":
		tm = tm.WithThinking(chat.ThinkOff)
	default:
		return nil, fmt.Errorf("SP0_THINK %q: want template, on or off", think)
	}
	return tk.EncodeSegments(tm.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
}

func sp0Equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sp0EnvStr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func sp0EnvInt(t *testing.T, k string, def int) int {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("%s=%q: %v", k, v, err)
	}
	return n
}

// sp0Suites is the registered prompt set: the verify-width sweep's code and math suites, and the copy-heavy edit suite.
func sp0Suites() map[string][]string {
	return map[string][]string{"code": widthSweepSuites["code"], "math": widthSweepSuites["math"], "edit": sp0EditSuite}
}

// sp0EditSuite is n-gram's traffic: a file in the prompt and a small change, with the whole file asked back.
var sp0EditSuite = []string{
	"Rename the function `add` to `sum` everywhere and return the whole file:\n\n```python\ndef add(a, b):\n    return a + b\n\n\ndef add_all(xs):\n    total = 0\n    for x in xs:\n        total = add(total, x)\n    return total\n\n\nif __name__ == \"__main__\":\n    print(add(2, 3))\n    print(add_all([1, 2, 3, 4]))\n```",
	"Add a docstring to every function and return the whole file:\n\n```go\npackage stack\n\ntype Stack struct {\n\titems []int\n}\n\nfunc (s *Stack) Push(v int) {\n\ts.items = append(s.items, v)\n}\n\nfunc (s *Stack) Pop() (int, bool) {\n\tif len(s.items) == 0 {\n\t\treturn 0, false\n\t}\n\tv := s.items[len(s.items)-1]\n\ts.items = s.items[:len(s.items)-1]\n\treturn v, true\n}\n\nfunc (s *Stack) Len() int {\n\treturn len(s.items)\n}\n```",
	"Change the log level from INFO to DEBUG in this config and return the whole file:\n\n```yaml\nserver:\n  host: 0.0.0.0\n  port: 8080\n  timeout_s: 30\nlogging:\n  level: INFO\n  format: json\n  output: stdout\ndatabase:\n  url: postgres://localhost:5432/app\n  pool_size: 10\n  timeout_s: 5\ncache:\n  enabled: true\n  ttl_s: 300\n```",
	"Make `fetch` retry up to 3 times on error, and return the whole file:\n\n```python\nimport requests\n\n\ndef fetch(url):\n    resp = requests.get(url, timeout=10)\n    resp.raise_for_status()\n    return resp.json()\n\n\ndef fetch_all(urls):\n    return [fetch(u) for u in urls]\n```",
	"Replace every `print` with `logger.info` and add `logger = logging.getLogger(__name__)` after the imports. Return the whole file:\n\n```python\nimport logging\nimport sys\n\n\ndef main(args):\n    print(\"starting\")\n    for a in args:\n        print(\"arg:\", a)\n    print(\"done\")\n    return 0\n\n\nif __name__ == \"__main__\":\n    sys.exit(main(sys.argv[1:]))\n```",
	"Add a `Clear` method that empties the cache, and return the whole file:\n\n```go\npackage cache\n\nimport \"sync\"\n\ntype Cache struct {\n\tmu sync.Mutex\n\tm  map[string]string\n}\n\nfunc New() *Cache {\n\treturn &Cache{m: map[string]string{}}\n}\n\nfunc (c *Cache) Get(k string) (string, bool) {\n\tc.mu.Lock()\n\tdefer c.mu.Unlock()\n\tv, ok := c.m[k]\n\treturn v, ok\n}\n\nfunc (c *Cache) Set(k, v string) {\n\tc.mu.Lock()\n\tdefer c.mu.Unlock()\n\tc.m[k] = v\n}\n```",
}
