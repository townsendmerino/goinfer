//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The GPU-resident columns of the option grid (decoder/optiongrid.go), driven on Metal. Each test sets
// the option through decoder.Options, loads the committed llama-tiny with decoder.Load, checks the
// option took hold and the model went Metal-resident, and runs the path through the entry point a
// caller uses (Model.Generate, Session.Generate, GenerateNgramSpeculative). Generation reports the
// chosen token's log-probability and the top five (SamplingParams.Logprobs), which serves as a numeric
// fingerprint: options that must not change the numbers are held to exact equality on it.
//
// These cover the Metal backend only. The grid has no backend axis, and a cell can hold on one GPU
// backend and not another (ActQuantGroup is honoured by CUDA residency and ignored by Metal's).

const mgFixture = "../testdata/llama-tiny"

// mgBase is the resident baseline every option is compared against: Metal runs int8 or int4 weights
// (an f32 load stays on the CPU).
var mgBase = decoder.Options{Backend: "metal", Quant: "int8int8"}

func mgWith(f func(*decoder.Options)) decoder.Options { o := mgBase; f(&o); return o }

func mgLoad(t *testing.T, opts decoder.Options) *decoder.Model {
	t.Helper()
	m, err := decoder.Load(mgFixture, opts)
	if err != nil {
		t.Fatalf("Load(%+v): %v", opts, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func mgResident(t *testing.T, m *decoder.Model) {
	t.Helper()
	if !strings.HasPrefix(m.DecodePath(), "metal-resident") {
		t.Fatalf("not Metal-resident: %s (decline: %s)", m.DecodePath(), m.ResidentDecline())
	}
}

// mgRun drains one generation and returns its tokens and its log-probability fingerprint.
func mgRun(t *testing.T, ch <-chan int, g *decoder.Generation) ([]int, []float64) {
	t.Helper()
	var toks []int
	for id := range ch {
		toks = append(toks, id)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generation: %v", err)
	}
	var lp []float64
	for _, s := range g.Logprobs {
		lp = append(lp, s.Logprob)
		for _, tl := range s.Top {
			lp = append(lp, tl.Logprob)
		}
	}
	return toks, lp
}

var mgSP = decoder.SamplingParams{Logprobs: true, TopLogprobs: 5}

func mgGen(t *testing.T, m *decoder.Model, prompt []int, n int) ([]int, []float64) {
	t.Helper()
	ch, g := m.Generate(context.Background(), prompt, n, mgSP)
	return mgRun(t, ch, g)
}

func mgSess(t *testing.T, s *decoder.Session, prompt []int, n int) ([]int, []float64) {
	t.Helper()
	ch, g := s.Generate(context.Background(), prompt, n, mgSP)
	return mgRun(t, ch, g)
}

func mgMaxDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i]))
	}
	return d
}

// mgShort is under every Metal batched-prefill floor, so after one sequential prefill everything is
// resident decode; mgLong is over them (16 tokens, 32 with two KV slots), so it takes the batched prefill.
func mgShort() []int { return []int{1, 17, 42, 99, 5, 17} }
func mgLong() []int {
	p := make([]int, 40)
	for i := range p {
		p[i] = (i*37 + 11) % 256
	}
	return p
}

// mgSpecPrompt repeats a phrase so the n-gram drafter proposes blocks.
func mgSpecPrompt() []int {
	return []int{1, 17, 42, 99, 5, 17, 42, 99, 5, 17, 42, 99, 5, 23, 61, 17, 42}
}

// mgRejecting proposes the known greedy continuation with each block's last token changed, so every
// verify accepts all but one token and must roll the rejected one back.
type mgRejecting struct {
	full   []int
	blocks int
}

func (r *mgRejecting) Draft(ctx []int, k int) []int {
	i := len(ctx)
	if k < 2 || i >= len(r.full) {
		return nil
	}
	out := append([]int(nil), r.full[i:min(i+k, len(r.full))]...)
	if len(out) < 2 {
		return nil
	}
	out[len(out)-1] ^= 1
	r.blocks++
	return out
}

// mgSpecMatchesPlain is the speculative-verify contract on one model: n-gram speculative decoding, with
// the production drafter and with one built to be partly rejected every round, emits exactly the
// tokens plain greedy decoding does on the same model.
func mgSpecMatchesPlain(t *testing.T, m *decoder.Model) {
	t.Helper()
	p := mgSpecPrompt()
	const n = 24
	ch, g := m.Generate(context.Background(), p, n, decoder.SamplingParams{})
	want, _ := mgRun(t, ch, g)
	full := append(append([]int(nil), p...), want...)
	for _, d := range []decoder.Drafter{&decoder.NgramDrafter{}, &mgRejecting{full: full}} {
		ch, g, err := m.GenerateNgramSpeculative(context.Background(), p, n, d, 4, decoder.SamplingParams{})
		if err != nil {
			t.Fatalf("%T: %v", d, err)
		}
		got, _ := mgRun(t, ch, g)
		if !slices.Equal(got, want) {
			t.Errorf("%T: speculative %v, plain %v", d, got, want)
		}
	}
}

// mgTwoTurns runs two turns on one Session (the second extends the first) and returns the second
// turn's tokens, its fingerprint, and how many prompt positions it reused.
func mgTwoTurns(t *testing.T, m *decoder.Model) ([]int, []float64, int) {
	t.Helper()
	s := m.NewSession(0)
	p := mgShort()
	first, _ := mgSess(t, s, p, 8)
	turn2 := append(append(append([]int(nil), p...), first...), 7, 11, 13)
	ch, g := s.Generate(context.Background(), turn2, 8, mgSP)
	toks, lp := mgRun(t, ch, g)
	return toks, lp, g.PrefillReused
}

// TestOptionPathMetal_neutralOptions: ResidentContext and ResidentKVSlots size the resident's KV
// allocation; neither may change a number. On resident decode, resident prefill, speculative verify and
// session reuse, a model loaded with the option must match the baseline exactly.
func TestOptionPathMetal_neutralOptions(t *testing.T) {
	ref := mgLoad(t, mgBase)
	mgResident(t, ref)
	refDecodeT, refDecodeL := mgGen(t, ref, mgShort(), 24)
	refPrefT, refPrefL := mgGen(t, ref, mgLong(), 4)
	refTurnT, refTurnL, refReused := mgTwoTurns(t, ref)
	if refReused == 0 {
		t.Fatal("the baseline session reused nothing on its second turn")
	}
	for _, c := range []struct {
		name   string
		opts   decoder.Options
		effect func(t *testing.T, m *decoder.Model)
	}{
		// 64, not more: llama-tiny's max_position_embeddings is 128, which already caps the default.
		{"ResidentContext 64", mgWith(func(o *decoder.Options) { o.ResidentContext = 64 }), func(t *testing.T, m *decoder.Model) {
			if got := m.ResidentContextCap(); got != 64 {
				t.Fatalf("ResidentContextCap %d, want 64", got)
			}
		}},
		{"ResidentKVSlots 2", mgWith(func(o *decoder.Options) { o.ResidentKVSlots = 2 }), func(t *testing.T, m *decoder.Model) {
			if got := m.ResidentKVSlots(); got != 2 {
				t.Fatalf("ResidentKVSlots %d, want 2", got)
			}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := mgLoad(t, c.opts)
			mgResident(t, m)
			c.effect(t, m)
			t.Run("resident decode", func(t *testing.T) {
				toks, lp := mgGen(t, m, mgShort(), 24)
				if !slices.Equal(toks, refDecodeT) || mgMaxDiff(lp, refDecodeL) != 0 {
					t.Errorf("differs from the baseline: %v vs %v, log-probabilities by %g", toks, refDecodeT, mgMaxDiff(lp, refDecodeL))
				}
			})
			t.Run("resident prefill", func(t *testing.T) {
				if batched, why := m.PrefillPath(); !batched {
					t.Fatalf("batched prefill not taken: %s", why)
				}
				toks, lp := mgGen(t, m, mgLong(), 4)
				if !slices.Equal(toks, refPrefT) || mgMaxDiff(lp, refPrefL) != 0 {
					t.Errorf("differs from the baseline: %v vs %v, log-probabilities by %g", toks, refPrefT, mgMaxDiff(lp, refPrefL))
				}
			})
			t.Run("speculative verify", func(t *testing.T) { mgSpecMatchesPlain(t, m) })
			t.Run("session reuse", func(t *testing.T) {
				toks, lp, reused := mgTwoTurns(t, m)
				if reused != refReused {
					t.Errorf("reused %d positions, the baseline %d", reused, refReused)
				}
				if !slices.Equal(toks, refTurnT) || mgMaxDiff(lp, refTurnL) != 0 {
					t.Errorf("second turn differs from the baseline: %v vs %v, log-probabilities by %g", toks, refTurnT, mgMaxDiff(lp, refTurnL))
				}
			})
		})
	}
}

// mgTeacherForced runs seq through m's resident one Forward at a time and returns every position's
// logits (copied: Forward's slice is reused).
func mgTeacherForced(t *testing.T, m *decoder.Model, seq []int) [][]float32 {
	t.Helper()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("no resident runner")
	}
	rf.Reset()
	out := make([][]float32, len(seq))
	for i, tok := range seq {
		l, err := rf.Forward(m.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident Forward[%d]: %v", i, err)
		}
		out[i] = append([]float32(nil), l...)
	}
	rf.Reset()
	return out
}

// mgCPUTeacherForced is the same on the CPU path.
func mgCPUTeacherForced(t *testing.T, m *decoder.Model, seq []int) [][]float32 {
	t.Helper()
	cache := m.NewCache(len(seq))
	out := make([][]float32, len(seq))
	for i, tok := range seq {
		l, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu Forward[%d]: %v", i, err)
		}
		out[i] = append([]float32(nil), l...)
	}
	return out
}

func mgMinCosine(a, b [][]float32) float64 {
	m := 1.0
	for i := range a {
		m = math.Min(m, cosineV(a[i], b[i]))
	}
	return m
}

// mgSeq is a fixed 32-token sequence both sides are forced through.
func mgSeq() []int {
	s := make([]int, 32)
	for i := range s {
		s[i] = (i*53 + 7) % 256
	}
	return s
}

// TestOptionPathMetal_quant: the resident decode and prefill cells of Quant. Metal's kernels are not
// the CPU's, so the contract is agreement, not identity: forced through the same tokens, every
// position's resident decode logits must stay within a cosine floor of the CPU's at the same Quant,
// and the batched prefill's first token must be the CPU's with its log-probabilities close.
func TestOptionPathMetal_quant(t *testing.T) {
	const decodeFloor, prefillTol = 0.999, 0.02
	for _, q := range []string{"int8int8", "int4"} {
		t.Run(q, func(t *testing.T) {
			g := mgLoad(t, mgWith(func(o *decoder.Options) { o.Quant = q }))
			mgResident(t, g)
			if !strings.Contains(g.DecodePath(), q) {
				t.Fatalf("resident at %s, asked for %s", g.DecodePath(), q)
			}
			c, err := decoder.Load(mgFixture, decoder.Options{Backend: "cpu", Quant: q})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			t.Run("resident decode", func(t *testing.T) {
				if cos := mgMinCosine(mgTeacherForced(t, g, mgSeq()), mgCPUTeacherForced(t, c, mgSeq())); cos < decodeFloor {
					t.Errorf("min per-position logit cosine %.6f vs the CPU at %s, floor %v", cos, q, decodeFloor)
				}
			})
			t.Run("resident prefill", func(t *testing.T) {
				if batched, why := g.PrefillPath(); !batched {
					t.Fatalf("batched prefill not taken: %s", why)
				}
				gt, gl := mgGen(t, g, mgLong(), 1)
				ct, cl := mgGen(t, c, mgLong(), 1)
				if !slices.Equal(gt, ct) || mgMaxDiff(gl, cl) > prefillTol {
					t.Errorf("resident prefill %v vs CPU %v, log-probabilities apart by %g (tolerance %v)", gt, ct, mgMaxDiff(gl, cl), prefillTol)
				}
			})
		})
	}
}

// TestOptionPathMetal_kvPrecision: KVPrecision's resident decode, speculative verify and session
// cells. Metal keeps its resident KV at f16 whatever is asked (ResidentKVPrecision says so), so "f16"
// must match the baseline exactly; "i8" is lossy, so its decode is held to a cosine floor against the
// f16 baseline, its speculative path to exact agreement with its own plain decoding, and its session
// reuse to agreement with a cold run of the same turn. Not exactness there: resident prefix reuse on
// Metal is not bit-identical to a cold prefill even at f16 (about 0.004 in log-probability on this
// model, kernels differing between decode and prefill), and resident_reuse.go does not claim it is.
func TestOptionPathMetal_kvPrecision(t *testing.T) {
	ref := mgLoad(t, mgBase)
	refT, refL := mgGen(t, ref, mgShort(), 24)
	t.Run("f16", func(t *testing.T) {
		m := mgLoad(t, mgWith(func(o *decoder.Options) { o.KVPrecision = "f16" }))
		mgResident(t, m)
		if p := m.ResidentKVPrecision(); p != "f16" {
			t.Fatalf("ResidentKVPrecision %q", p)
		}
		toks, lp := mgGen(t, m, mgShort(), 24)
		if !slices.Equal(toks, refT) || mgMaxDiff(lp, refL) != 0 {
			t.Errorf("KVPrecision f16 differs from Metal's default (f16): log-probabilities by %g", mgMaxDiff(lp, refL))
		}
		mgSpecMatchesPlain(t, m)
	})
	t.Run("i8", func(t *testing.T) {
		m := mgLoad(t, mgWith(func(o *decoder.Options) { o.KVPrecision = "i8" }))
		mgResident(t, m)
		if p := m.ResidentKVPrecision(); p != "i8" {
			t.Fatalf("ResidentKVPrecision %q", p)
		}
		t.Run("resident decode", func(t *testing.T) {
			const floor = 0.999
			got, want := mgTeacherForced(t, m, mgSeq()), mgTeacherForced(t, ref, mgSeq())
			cos := mgMinCosine(got, want)
			if cos < floor {
				t.Errorf("min per-position logit cosine %.6f vs f16 KV, floor %v", cos, floor)
			}
			if cos == 1 {
				t.Error("int8 KV gives the f16 logits exactly: the option did not take hold")
			}
		})
		t.Run("speculative verify", func(t *testing.T) { mgSpecMatchesPlain(t, m) })
		t.Run("session reuse", func(t *testing.T) {
			toks, lp, reused := mgTwoTurns(t, m)
			if reused == 0 {
				t.Fatal("the second turn reused nothing")
			}
			s := m.NewSession(0)
			p := mgShort()
			first, _ := mgSess(t, s, p, 8)
			turn2 := append(append(append([]int(nil), p...), first...), 7, 11, 13)
			coldT, coldL := mgGen(t, m, turn2, 8)
			const tol = 0.02
			if !slices.Equal(toks, coldT) || mgMaxDiff(lp, coldL) > tol {
				t.Errorf("reused second turn %v, cold %v, log-probabilities apart by %g (tolerance %v)", toks, coldT, mgMaxDiff(lp, coldL), tol)
			}
		})
	})
}

// TestOptionPathMetal_exactPrefill: with ExactPrefill the resident prefill must be the sequential path's
// result exactly, which a knob-forced sequential prefill on the baseline gives, and must differ from the
// baseline's batched f16 prefill (or the case is not exercising the option).
func TestOptionPathMetal_exactPrefill(t *testing.T) {
	m := mgLoad(t, mgWith(func(o *decoder.Options) { o.ExactPrefill = true }))
	mgResident(t, m)
	if batched, _ := m.PrefillPath(); batched {
		t.Fatal("ExactPrefill left the batched prefill on")
	}
	seq := mgLoad(t, mgWith(func(o *decoder.Options) {
		o.Knobs = &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL": "0"}
	}))
	fast := mgLoad(t, mgBase)
	gt, gl := mgGen(t, m, mgLong(), 4)
	st, sl := mgGen(t, seq, mgLong(), 4)
	_, fl := mgGen(t, fast, mgLong(), 4)
	if !slices.Equal(gt, st) || mgMaxDiff(gl, sl) != 0 {
		t.Errorf("ExactPrefill %v vs sequential %v, log-probabilities apart by %g", gt, st, mgMaxDiff(gl, sl))
	}
	if mgMaxDiff(gl, fl) == 0 {
		t.Error("ExactPrefill matches the batched f16 prefill exactly: the case does not tell the paths apart")
	}
}

// TestOptionPathMetal_prefillChunk: ResidentPrefillChunk under MC3 prefills a long prompt in chunks while
// another generation is decoding. Metal's batched prefill is chunk-invariant, so the chunked generation
// must match the same generation prefilled whole, and the batcher's prefill-pass count must show the
// chunks ran.
//
// The chunk is 32, the batched-prefill floor with two KV slots. Below the floor the invariance does not
// hold: at chunk 8 or 16 the chunked reply differs from the whole one by 0.003-0.004 in log-probability
// on this model (2026-10-08), so a reply then depends on whether another generation was decoding.
// serve's default chunk (512) is far above every floor; the finding is recorded in
// docs/tasks/task-option-path-admission-2026-10.md.
func TestOptionPathMetal_prefillChunk(t *testing.T) {
	opts := mgWith(func(o *decoder.Options) { o.ResidentKVSlots = 2; o.ResidentPrefillChunk = 32 })
	m := mgLoad(t, opts)
	mgResident(t, m)
	if n := m.EnableResidentConcurrency(2); n != 2 {
		t.Fatalf("EnableResidentConcurrency(2) = %d", n)
	}
	// The reference runs on its own model: the resident remembers a committed prefix across generations,
	// so running the same prompt first on m would leave the measured run nothing to prefill.
	ref := mgLoad(t, opts)

	// Chunking happens only while another generation is inside its decode loop, so a second generation
	// decodes (read continuously) while this one prefills. Whether they overlap is up to the scheduler, so
	// attempts repeat, each with a fresh 96-token prompt, until the pass count shows chunks; the output is
	// checked on every attempt.
	const attempts = 5
	for a := 1; ; a++ {
		long := make([]int, 96)
		for i := range long {
			long[i] = (i*37 + 11*a + 3) % 256
		}
		wantT, wantL := mgGen(t, ref, long, 4) // alone: nobody decoding, so prefilled whole

		before := m.ResidentBatchStats().PrefillPasses
		aCh, _ := m.Generate(context.Background(), []int{1, 2, 3, 5 + a}, 100, decoder.SamplingParams{})
		<-aCh
		done := make(chan struct{})
		go func() {
			for range aCh {
			}
			close(done)
		}()
		gotT, gotL := mgGen(t, m, long, 4)
		passes := m.ResidentBatchStats().PrefillPasses - before
		<-done
		if !slices.Equal(gotT, wantT) || mgMaxDiff(gotL, wantL) != 0 {
			t.Errorf("attempt %d: chunked %v vs whole %v, log-probabilities apart by %g", a, gotT, wantT, mgMaxDiff(gotL, wantL))
		}
		if passes >= 4 { // the other generation's prefill, two 32-token chunks, and the final pass
			break
		}
		if a == attempts {
			t.Fatalf("%d prefill passes on the last of %d attempts: the prompt was never chunked", passes, attempts)
		}
	}
}

// TestOptionPathMetal_embedInt4Declines: Metal's resident needs an int8 embedding table (int8Buf), so
// int4 with EmbedInt4 is declined at load and runs on the CPU, and its output must be the CPU's.
func TestOptionPathMetal_embedInt4Declines(t *testing.T) {
	m := mgLoad(t, decoder.Options{Backend: "metal", Quant: "int4", EmbedInt4: true})
	if strings.HasPrefix(m.DecodePath(), "metal-resident") {
		t.Fatal("int4 with EmbedInt4 went Metal-resident")
	}
	if !strings.Contains(m.ResidentDecline(), "is not int8") {
		t.Fatalf("decline does not name the int8 embedding requirement: %s", m.ResidentDecline())
	}
	c, err := decoder.Load(mgFixture, decoder.Options{Backend: "cpu", Quant: "int4", EmbedInt4: true})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for _, p := range [][]int{mgShort(), mgLong()} {
		gt, gl := mgGen(t, m, p, 12)
		ct, cl := mgGen(t, c, p, 12)
		if !slices.Equal(gt, ct) || mgMaxDiff(gl, cl) != 0 {
			t.Errorf("declined load %v, CPU %v, log-probabilities apart by %g", gt, ct, mgMaxDiff(gl, cl))
		}
	}
}
