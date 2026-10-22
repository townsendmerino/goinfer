//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// KV4a's storage-identity gates (docs/measurements/kv4-cuda-2026-10-21.md §2, G1a and G1c): paged storage gives the same bits whatever
// physical blocks a sequence's table names, and a windowed layer read through a table that has released its oldest blocks gives the same
// bits as the full cache.

// kvScrambledOrder hands a pool's blocks out reversed and then interleaved from both ends, so logically consecutive blocks of a slot are
// physically far apart and out of order, and two slots' blocks interleave.
func kvScrambledOrder(n int) []int32 {
	rev := make([]int32, 0, n-1)
	for b := n - 1; b >= 1; b-- {
		rev = append(rev, int32(b))
	}
	out := make([]int32, 0, len(rev))
	for i, j := 0, len(rev)-1; i <= j; i, j = i+1, j-1 {
		out = append(out, rev[i])
		if i != j {
			out = append(out, rev[j])
		}
	}
	return out
}

// kv4Trace is every logits vector a workload produced, in order, with a label for the step that produced it.
type kv4Trace struct {
	steps  []string
	logits [][]float32
}

func (tr *kv4Trace) add(step string, lg []float32) {
	tr.steps = append(tr.steps, step)
	tr.logits = append(tr.logits, append([]float32(nil), lg...))
}

// kv4Workload drives one resident through every KV path a table must serve: a batched prefill (or per-token where the fixture declines
// it), decode, a continuing turn, a 4-row verify, a second slot, an MC3 step over both slots where the resident has one, and decode back
// on the first slot. Token ids are fixed, never sampled, so two residents see the same inputs.
func kv4Workload(t *testing.T, m *decoder.Model, r *cudaResident, prompt int) kv4Trace {
	t.Helper()
	_, _, _, _, _, _, vocab := m.Dims()
	id := func(i, salt int) int { return (i*31 + salt*17 + 7) % vocab }
	emb := func(i, salt int) []float32 { return append([]float32(nil), m.EmbedResidentForTest(id(i, salt))...) }
	var tr kv4Trace
	prefill := func(label string, n, start, salt int) {
		e := make([][]float32, n)
		for i := range e {
			e[i] = emb(start+i, salt)
		}
		lg, err := r.PrefillLast(context.Background(), e, start)
		if err == nil {
			tr.add(label, lg)
			return
		}
		if !errors.Is(err, errPrefillDeclined) {
			t.Fatalf("%s: PrefillLast(%d rows at %d): %v", label, n, start, err)
		}
		for i := range e {
			lg, err := r.Forward(e[i], start+i)
			if err != nil {
				t.Fatalf("%s: Forward at %d: %v", label, start+i, err)
			}
			if i == len(e)-1 {
				tr.add(label, lg)
			}
		}
	}
	decode := func(label string, from, n, salt int) {
		for i := range n {
			lg, err := r.Forward(emb(from+i, salt), from+i)
			if err != nil {
				t.Fatalf("%s: Forward at %d: %v", label, from+i, err)
			}
			tr.add(label, lg)
		}
	}
	bind := func(s int) {
		if err := r.UseKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	bind(0)
	prefill("prompt", prompt, 0, 1)
	pos := prompt
	decode("decode", pos, 64, 2)
	pos += 64
	prefill("turn", 40, pos, 3)
	pos += 40
	ve := make([][]float32, 4)
	for i := range ve {
		ve[i] = emb(pos+i, 4)
	}
	rows, err := r.ForwardN(ve, pos)
	if err != nil {
		t.Fatalf("verify at %d: %v", pos, err)
	}
	for _, lg := range rows {
		tr.add("verify", lg)
	}
	if r.KVSlots() > 1 {
		bind(1)
		prefill("slot 1 prompt", prompt/2, 0, 5)
		pos1 := prompt / 2
		decode("slot 1 decode", pos1, 8, 6)
		pos1 += 8
		if lo, hi := r.BatchStepRange(); hi >= 2 && lo <= 2 {
			for step := range 3 {
				out, err := r.StepBatch([]decoder.ResidentBatchSeq{
					{Slot: 0, Pos: pos + step, Emb: emb(pos+step, 7)},
					{Slot: 1, Pos: pos1 + step, Emb: emb(pos1+step, 8)},
				})
				if err != nil {
					t.Fatalf("MC3 step %d: %v", step, err)
				}
				for _, o := range out {
					tr.add("mc3", o.Logits)
				}
			}
			pos += 3
		}
		bind(0)
	}
	decode("decode again", pos, 8, 9)
	return tr
}

// kv4SameTrace fails t at the first step whose logits differ by a bit.
func kv4SameTrace(t *testing.T, want, got kv4Trace) {
	t.Helper()
	if len(got.logits) != len(want.logits) {
		t.Fatalf("%d steps, want %d", len(got.logits), len(want.logits))
	}
	for i := range want.logits {
		if j, ok := sameBits(got.logits[i], want.logits[i]); !ok {
			t.Fatalf("step %d (%s): logit %d differs: %v vs %v", i, want.steps[i], j, got.logits[i][j], want.logits[i][j])
		}
	}
}

// TestKV4_scrambledTables is G1a: on every tiny fixture the CUDA resident loads, the workload's every logit is bit-identical between a
// pool that hands blocks out in ascending order and one that hands them out scrambled (kvScrambledOrder), at f32 and at f16 where the
// family takes it. The windowed fixtures run with a small slack, so their window group releases and re-takes blocks many times. A "+flash"
// or "+splitkv" entry runs its fixture with that decode lane forced on from the first key, and its launch counter must move.
func TestKV4_scrambledTables(t *testing.T) {
	requireCUDADevice(t)
	windowed := map[string]bool{"mistral-tiny-window": true, "gemma3-vl-tiny": true, "cohere2-tiny": true, "gemma2-tiny": true}
	for _, fx := range []string{
		"llama-tiny", "llama-attnfa-tiny", "mistral-tiny-window", "gemma3-vl-tiny", "cohere2-tiny", "gemma2-tiny", "cohere-tiny",
		"qwen25vl-tiny", "qwen3moe-tiny", "deepseek-tiny", "qwen35-tiny", "gemma4-moe-kv-tiny32",
		"llama-attnfa-tiny+flash", "llama-attnfa-tiny+splitkv",
	} {
		for _, kv := range []string{"f32", "f16"} {
			t.Run(fx+"/kv="+kv, func(t *testing.T) {
				fx, lane, _ := strings.Cut(fx, "+")
				if _, err := os.Stat(filepath.Join("..", "testdata", fx)); err != nil {
					t.Skipf("no fixture: %v", err)
				}
				opts := decoder.Options{ResidentKVSlots: 2, KVPrecision: kv}
				switch lane {
				case "flash":
					opts.Knobs = &decoder.Knobs{"GOINFER_CUDA_FLASH_DECODE": "16", "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS": "0"}
				case "splitkv":
					opts.Knobs = &decoder.Knobs{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_SPLITKV_MIN_KEYS": "0"}
				}
				if windowed[fx] {
					withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, windowedKVTestSlack-windowedKVTestStep)
					opts.ResidentWindowedKV = true
				}
				if kv == "f16" {
					if why := kv4F16Refusal(t, fx, opts); why != "" {
						t.Skipf("f16 is not delivered for this family yet (KV4b): %s", why)
					}
				}
				run := func(order func(int) []int32) (kv4Trace, *cudaResident) {
					old := kvPoolOrderForTest
					kvPoolOrderForTest = order
					defer func() { kvPoolOrderForTest = old }()
					m, r := loadResident(t, fx, opts)
					return kv4Workload(t, m, r, min(600, r.ctxCap/2)), r
				}
				want, _ := run(nil)
				got, r := run(kvScrambledOrder)
				if windowed[fx] && !r.kvWin {
					t.Fatalf("%s: windowed KV did not engage (context %d)", fx, r.ctxCap)
				}
				t.Logf("%d steps, %d slots, windowed %v, f16 %v; flash-decode launches %d, split-KV %d", len(want.logits), r.KVSlots(), r.kvWin,
					r.kvF16, r.faLaunches, r.skLaunches)
				if lane == "flash" && r.faLaunches == 0 || lane == "splitkv" && r.skLaunches == 0 {
					t.Fatalf("VOID: the %s lane never launched", lane)
				}
				kv4SameTrace(t, want, got)
			})
		}
	}
}

// TestKV4_scrambledTablesCanFail is G1a's control: a pool that hands out one block for every request (so every position of every slot shares
// it) must change the trace, or G1a could not see a table that names the wrong block.
func TestKV4_scrambledTablesCanFail(t *testing.T) {
	requireCUDADevice(t)
	opts := decoder.Options{ResidentKVSlots: 2, KVPrecision: "f32"}
	m, r := loadResident(t, "llama-tiny", opts)
	want := kv4Workload(t, m, r, min(600, r.ctxCap/2))
	old := kvPoolOrderForTest
	kvPoolOrderForTest = func(n int) []int32 {
		out := make([]int32, n-1)
		for i := range out {
			out[i] = 1
		}
		return out
	}
	m2, r2 := loadResident(t, "llama-tiny", opts)
	kvPoolOrderForTest = old
	got := kv4Workload(t, m2, r2, min(600, r2.ctxCap/2))
	for i := range want.logits {
		if _, ok := sameBits(got.logits[i], want.logits[i]); !ok {
			t.Logf("the planted defect shows at step %d (%s)", i, want.steps[i])
			return
		}
	}
	t.Fatal("a pool whose every block is the same block gave the same trace: the workload cannot see a table defect")
}

// kv4F16Refusal is why a load of fx under opts at --kv f16 is refused, "" when it is delivered.
func kv4F16Refusal(t *testing.T, fx string, opts decoder.Options) string {
	t.Helper()
	opts.Backend, opts.Quant, opts.KVPrecision = "cuda", "int4", "f16"
	m, err := decoder.Load(filepath.Join("..", "testdata", fx), opts)
	if err != nil {
		var r *decoder.KVRefusal
		if errors.As(err, &r) {
			return err.Error()
		}
		t.Fatalf("load %s at f16: %v", fx, err)
	}
	_ = m.Close()
	return ""
}

// kv4WindowFixture writes a copy of llama-attnfa-tiny (head dim 128, so the flash-decode lane serves it) as a Mistral with a sliding window
// of window positions and a context of maxPos, and returns its directory.
func kv4WindowFixture(t *testing.T, window, maxPos int) string {
	t.Helper()
	src := filepath.Join("..", "testdata", "llama-attnfa-tiny")
	raw, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["architectures"] = []string{"MistralForCausalLM"}
	cfg["model_type"] = "mistral"
	cfg["sliding_window"] = window
	cfg["max_position_embeddings"] = maxPos
	dir := t.TempDir()
	out, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), out, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"model.safetensors", "generation_config.json"} {
		abs, err := filepath.Abs(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(abs, filepath.Join(dir, f)); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestKV4_windowedFlashAndSplitKV is G1c: flash-decode and split-KV read a windowed layer through a table whose blocks behind the window
// have been released. A 4,096-position window, a 9,000-token prompt and 64 decoded tokens, so every decoded row attends 4,096 keys
// (both lanes' floors) and the window group has released blocks behind its base; each lane's launch counter must move, or the arm is
// void. Pass: every decoded logit bit-identical to the same build with windowed KV off, where a sliding-window layer reads the same
// [p - W + 1, p] from the full cache. Run at f32 and f16 KV.
func TestKV4_windowedFlashAndSplitKV(t *testing.T) {
	requireCUDADevice(t)
	const window, maxPos, prompt, steps = 4096, 16384, 9000, 64
	dir := kv4WindowFixture(t, window, maxPos)
	for _, kv := range []string{"f32", "f16"} {
		for _, lane := range []string{"flash", "splitkv"} {
			t.Run(kv+"/"+lane, func(t *testing.T) {
				knobs := decoder.Knobs{"GOINFER_CUDA_FLASH_DECODE": "16"} // the suite pins the lane off (TestMain)
				if lane == "splitkv" {
					knobs["GOINFER_CUDA_FLASH_DECODE"] = "0"
					knobs["GOINFER_SPLITKV_MIN_KEYS"] = "0"
				}
				trace := func(win bool) (kv4Trace, *cudaResident) {
					m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: kv, ResidentContext: maxPos,
						ResidentWindowedKV: win, Knobs: &knobs})
					var refused *decoder.KVRefusal
					if errors.As(err, &refused) && kv == "f16" {
						t.Skipf("f16 windowed KV is KV4b's: %v", err)
					}
					if err != nil {
						t.Fatalf("load: %v", err)
					}
					t.Cleanup(func() { _ = m.Close() })
					r := mustResident(t, m)
					if win != r.kvWin {
						t.Fatalf("windowed KV %v asked, engaged %v", win, r.kvWin)
					}
					_, _, _, _, _, _, vocab := m.Dims()
					e := make([][]float32, prompt)
					for i := range e {
						e[i] = append([]float32(nil), m.EmbedResidentForTest((i*31+7)%vocab)...)
					}
					var tr kv4Trace
					lg, err := r.PrefillLast(context.Background(), e, 0)
					if err != nil {
						t.Fatalf("prefill: %v", err)
					}
					tr.add("prompt", lg)
					fa, sk := r.faLaunches, r.skLaunches
					for i := range steps {
						lg, err := r.Forward(m.EmbedResidentForTest((i*13+5)%vocab), prompt+i)
						if err != nil {
							t.Fatalf("decode at %d: %v", prompt+i, err)
						}
						tr.add("decode", lg)
					}
					if lane == "flash" && r.faLaunches == fa {
						t.Fatalf("VOID: windowed %v decoded %d rows past %d keys without a flash-decode launch", win, steps, window)
					}
					if lane == "splitkv" && r.skLaunches == sk {
						t.Fatalf("VOID: windowed %v decoded %d rows past %d keys without a split-KV pass", win, steps, window)
					}
					return tr, r
				}
				want, _ := trace(false)
				got, r := trace(true)
				released := 0
				for _, id := range r.kvTabs[r.kvSlot].ids[kvGroupWin][:r.kvBases[r.kvSlot]/kvBlock] {
					if id == 0 {
						released++
					}
				}
				if released == 0 {
					t.Fatalf("VOID: the window group released no block behind its base %d", r.kvBases[r.kvSlot])
				}
				t.Logf("base %d, %d blocks behind it released; f16 %v", r.kvBases[r.kvSlot], released, r.kvF16)
				kv4SameTrace(t, want, got)
			})
		}
	}
}
