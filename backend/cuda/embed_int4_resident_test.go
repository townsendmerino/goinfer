//go:build cuda && goinfer_testhooks

package cuda

import (
	"bufio"
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fidelity"
)

// Options.EmbedInt4 reaches every family loader (decoder/embed_int4_loaders_test.go), so a family that used to get an int8 head by default
// now gets an int4 one. The resident must take that table without changing its decision: where a fixture builds resident with the int8 head it builds resident
// with the int4 head (the same decode path, the head reading int4), and where it declines it declines for the same reason either way. A resident that
// declined an int4 head would silently move a whole family to the CPU the day the default reached it.
func TestEmbedInt4_residentDecisionDoesNotChange(t *testing.T) {
	requireCUDADevice(t)
	residentSeen := 0
	for _, d := range []string{"llama-tiny", "granite-dense-tiny", "phi3-tiny", "glm-ocr-tiny", "granite-tiny", "nemotron-tiny", "nemotron3nano-tiny", "llama4-tiny", "gpt2"} {
		t.Run(d, func(t *testing.T) {
			dir := filepath.Join("../../testdata", d)
			var path, head [2]string
			for i, on := range []bool{false, true} {
				m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", EmbedInt4: on, ResidentContext: 64})
				if err != nil {
					t.Skipf("EmbedInt4=%v: %v", on, err)
				}
				path[i], head[i] = m.DecodePath(), m.HeadTable()
				m.Close()
			}
			if head[0] != "int8" || head[1] != "int4" {
				t.Errorf("head tables %q / %q, want int8 / int4", head[0], head[1])
			}
			if path[0] != path[1] {
				t.Errorf("the decode path changed with the head table:\n  int8 head: %s\n  int4 head: %s", path[0], path[1])
			}
			if strings.HasPrefix(path[1], "cuda-resident") {
				residentSeen++
			}
		})
	}
	if residentSeen < 2 {
		t.Errorf("only %d of the fixtures built resident: the test is not looking at a resident that takes an int4 head", residentSeen)
	}
}

// TestEmbedInt4LongContext_run is #1160's run harness (docs/measurements/embed-int4-longctx-2026-11-20/README.md): one arm, teacher-forced
// over the reference's documents, scoring each registered position against the reference's log-probabilities. Without EI4LC_ARM it skips.
// EI4LC_ARM is cuda-int4head, cuda-int8head or cpu-int4head; EI4LC_MODEL the checkpoint; EI4LC_REF the reference directory (ref.py's
// output); EI4LC_OUT the JSONL to write; EI4LC_DOCS optionally limits the documents (comma-separated names).
func TestEmbedInt4LongContext_run(t *testing.T) {
	arm, model, refDir, out := os.Getenv("EI4LC_ARM"), os.Getenv("EI4LC_MODEL"), os.Getenv("EI4LC_REF"), os.Getenv("EI4LC_OUT")
	if arm == "" {
		t.Skip("EI4LC_ARM unset: this is a run harness, not a gate")
	}
	opts := decoder.Options{Quant: "int4"}
	switch arm {
	case "cuda-int4head", "cuda-int8head":
		requireCUDADevice(t)
		opts.Backend, opts.EmbedInt4, opts.ResidentContext, opts.KVPrecision = "cuda", arm == "cuda-int4head", 8320, "f16"
	case "cpu-int4head":
		opts.Backend, opts.EmbedInt4 = "cpu", true
	default:
		t.Fatalf("EI4LC_ARM=%q", arm)
	}
	docs, err := ei4lcReadRefs(refDir, os.Getenv("EI4LC_DOCS"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(model, opts)
	if err != nil {
		t.Fatalf("load %s: %v", arm, err)
	}
	defer m.Close()
	wantHead := map[bool]string{true: "int4", false: "int8"}[opts.EmbedInt4]
	if m.HeadTable() != wantHead {
		t.Fatalf("head table %q, want %q", m.HeadTable(), wantHead)
	}
	var rf decoder.ResidentForward
	if opts.Backend == "cuda" {
		if rf = m.ResidentForwardForTest(); rf == nil || !strings.HasPrefix(m.DecodePath(), "cuda-resident") {
			t.Fatalf("VOID: not resident on CUDA (%s; %s)", m.DecodePath(), m.ResidentDecline())
		}
	}
	fmt.Fprintf(os.Stderr, "ei4lc %s: %s, head %s, %d documents\n", arm, m.DecodePath(), m.HeadTable(), len(docs))
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	start, beat := time.Now(), time.Now()
	for di, d := range docs {
		var cache *decoder.KVCache
		if rf != nil {
			rf.Reset()
		} else {
			cache = m.NewCache(len(d.IDs) + 1)
		}
		last := d.Scored[len(d.Scored)-1]
		si := 0
		for p := 0; p <= last; p++ {
			var lg []float32
			if rf != nil {
				lg, err = rf.Forward(m.EmbedResidentForTest(d.IDs[p]), p)
			} else {
				lg, err = m.ForwardForTest(d.IDs[p], cache)
			}
			if err != nil {
				t.Fatalf("%s pos %d: %v", d.Name, p, err)
			}
			if si < len(d.Scored) && d.Scored[si] == p {
				ref := d.LP[si*d.Vocab : (si+1)*d.Vocab]
				if len(lg) < d.Vocab {
					t.Fatalf("%s: %d logits, reference vocab %d", d.Name, len(lg), d.Vocab)
				}
				row := ei4lcScore(ref, lg[:d.Vocab])
				row.Doc, row.K, row.Pos = d.Name, d.KOf[si], p
				if err := enc.Encode(row); err != nil {
					t.Fatal(err)
				}
				si++
			}
			if time.Since(beat) > 30*time.Second {
				beat = time.Now()
				fmt.Fprintf(os.Stderr, "ei4lc %s: doc %d/%d %s pos %d/%d, elapsed %s\n", arm, di+1, len(docs), d.Name, p, last, time.Since(start).Round(time.Second))
			}
		}
		fmt.Fprintf(os.Stderr, "ei4lc %s: doc %d/%d %s done, elapsed %s\n", arm, di+1, len(docs), d.Name, time.Since(start).Round(time.Second))
	}
}

// TestEmbedInt4LongContext_grade grades the run harness's arms by the registration: fidelity.NonInferior with the int4 head as the candidate
// and the int8 head as the exact arm, cells by K, each (document, K) window a prompt. EI4LC_GRADE_DIR holds <arm>.jsonl per model as
// <model>-<arm>.jsonl; EI4LC_GRADE_MODELS names the models. Without EI4LC_GRADE_DIR it skips.
func TestEmbedInt4LongContext_grade(t *testing.T) {
	dir := os.Getenv("EI4LC_GRADE_DIR")
	if dir == "" {
		t.Skip("EI4LC_GRADE_DIR unset")
	}
	crit := fidelity.Criteria{AgreeMarginPts: 2.5, KLMargin: 0.25, Alpha: 0.05}
	for model := range strings.SplitSeq(os.Getenv("EI4LC_GRADE_MODELS"), ",") {
		arms := map[string]map[string][]ei4lcRow{}
		for _, a := range []string{"cuda-int4head", "cuda-int8head", "cpu-int4head"} {
			rows, err := ei4lcReadRows(filepath.Join(dir, model+"-"+a+".jsonl"))
			if err != nil {
				t.Logf("%s %s: %v", model, a, err)
				continue
			}
			arms[a] = rows
			ei4lcLogArm(t, model, a, rows)
		}
		cand, exact := arms["cuda-int4head"], arms["cuda-int8head"]
		if cand == nil || exact == nil {
			t.Errorf("%s: both CUDA arms are needed to grade", model)
			continue
		}
		byK := map[int][]fidelity.Prompt{}
		for key, c := range cand {
			e, ok := exact[key]
			if !ok || len(e) != len(c) {
				t.Fatalf("%s: window %s differs between arms", model, key)
			}
			pc, pe := make([]fidelity.Position, len(c)), make([]fidelity.Position, len(e))
			for i := range c {
				if c[i].Pos != e[i].Pos {
					t.Fatalf("%s %s: positions differ", model, key)
				}
				pc[i] = fidelity.Position{Agree: c[i].Agree, HardFlip: c[i].Flip, KL: c[i].KL}
				pe[i] = fidelity.Position{Agree: e[i].Agree, HardFlip: e[i].Flip, KL: e[i].KL}
			}
			byK[c[0].K] = append(byK[c[0].K], fidelity.Prompt{ID: key, Candidate: pc, Exact: pe})
		}
		var cells []fidelity.Cell
		var ks []int
		for k := range byK {
			ks = append(ks, k)
		}
		slices.Sort(ks)
		for _, k := range ks {
			slices.SortFunc(byK[k], func(a, b fidelity.Prompt) int { return cmp.Compare(a.ID, b.ID) })
			cells = append(cells, fidelity.Cell{Name: "K" + strconv.Itoa(k), Prompts: byK[k]})
		}
		v, err := fidelity.NonInferior(cells, crit)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.MarshalIndent(v, "", " ")
		_ = os.WriteFile(filepath.Join(dir, model+"-verdict.json"), b, 0o644)
		p := v.Pooled
		t.Logf("%s VERDICT pass=%v (agree %v, KL %v, flips %v) pooled: agree int4 %.2f%% int8 %.2f%%, diff %.2f pts (LCB %.2f), KL ratio %.3f (UCB %.3f), flips %d/%d, %d prompts %d positions; %v",
			model, v.Pass, v.AgreePass, v.KLPass, v.FlipsPass, p.AgreeCandPct, p.AgreeExactPct, p.AgreeDiffPts, p.AgreeDiffLCB, p.KLRatio, p.KLRatioUCB, p.FlipsCand, p.FlipsExact, p.Prompts, p.Positions, v.Reasons)
		for _, k := range ks {
			s := v.PerCell["K"+strconv.Itoa(k)]
			t.Logf("%s K=%d: agree int4 %.2f%% int8 %.2f%% (diff %.2f pts), mean KL int4 %.4f int8 %.4f (ratio %.3f), flips %d/%d",
				model, k, s.AgreeCandPct, s.AgreeExactPct, s.AgreeDiffPts, s.MeanKLCand, s.MeanKLExact, s.KLRatio, s.FlipsCand, s.FlipsExact)
		}
	}
}

type ei4lcDoc struct {
	Name   string
	IDs    []int
	Scored []int
	KOf    []int
	Vocab  int
	LP     []float32
}

type ei4lcRow struct {
	Doc    string  `json:"doc"`
	K      int     `json:"k"`
	Pos    int     `json:"pos"`
	Agree  bool    `json:"agree"`
	Flip   bool    `json:"flip"`
	KL     float64 `json:"kl"`
	RefTop int     `json:"ref_top"`
	ArmTop int     `json:"arm_top"`
}

// ei4lcScore scores one position: KL(reference ‖ arm) in nats over the full vocabulary, top-1 agreement, and the near-tie hard flip.
func ei4lcScore(refLP, logits []float32) ei4lcRow {
	mx := math.Inf(-1)
	for _, v := range logits {
		mx = math.Max(mx, float64(v))
	}
	var z float64
	for _, v := range logits {
		z += math.Exp(float64(v) - mx)
	}
	lz := mx + math.Log(z)
	var kl float64
	rt, at := 0, 0
	for i, r := range refLP {
		if r > refLP[rt] {
			rt = i
		}
		if logits[i] > logits[at] {
			at = i
		}
		if pr := math.Exp(float64(r)); pr > 0 {
			kl += pr * (float64(r) - (float64(logits[i]) - lz))
		}
	}
	_, _, hard := decoder.NearTieArgmaxForTest(refLP, logits)
	return ei4lcRow{Agree: rt == at, Flip: hard, KL: kl, RefTop: rt, ArmTop: at}
}

func ei4lcReadRefs(dir, only string) ([]ei4lcDoc, error) {
	names, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil || len(names) == 0 {
		return nil, fmt.Errorf("no reference documents in %q (%v)", dir, err)
	}
	slices.Sort(names)
	var docs []ei4lcDoc
	for _, n := range names {
		var h struct {
			Doc    string `json:"doc"`
			IDs    []int  `json:"ids"`
			Scored []int  `json:"scored"`
			KOf    []int  `json:"k_of"`
			Vocab  int    `json:"vocab"`
		}
		b, err := os.ReadFile(n)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, &h); err != nil {
			return nil, fmt.Errorf("%s: %v", n, err)
		}
		if only != "" && !strings.Contains(","+only+",", ","+h.Doc+",") {
			continue
		}
		raw, err := os.ReadFile(strings.TrimSuffix(n, ".json") + ".lp.f16")
		if err != nil {
			return nil, err
		}
		if len(raw) != 2*len(h.Scored)*h.Vocab {
			return nil, fmt.Errorf("%s: %d bytes of log-probs, want %d", n, len(raw), 2*len(h.Scored)*h.Vocab)
		}
		lp := make([]float32, len(raw)/2)
		for i := range lp {
			lp[i] = ei4lcHalf(uint16(raw[2*i]) | uint16(raw[2*i+1])<<8)
		}
		docs = append(docs, ei4lcDoc{Name: h.Doc, IDs: h.IDs, Scored: h.Scored, KOf: h.KOf, Vocab: h.Vocab, LP: lp})
	}
	return docs, nil
}

// ei4lcHalf widens an IEEE 754 binary16 to float32.
func ei4lcHalf(h uint16) float32 {
	sign, exp, frac := uint32(h>>15)<<31, uint32(h>>10)&0x1f, uint32(h)&0x3ff
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal: frac × 2^-24
		v := float32(frac) / (1 << 24)
		if sign != 0 {
			v = -v
		}
		return v
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
}

func ei4lcReadRows(path string) (map[string][]ei4lcRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string][]ei4lcRow{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r ei4lcRow
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, err
		}
		key := r.Doc + "@" + strconv.Itoa(r.K)
		out[key] = append(out[key], r)
	}
	return out, sc.Err()
}

func ei4lcLogArm(t *testing.T, model, arm string, rows map[string][]ei4lcRow) {
	type acc struct {
		n, agree, flips int
		kl              float64
	}
	byK := map[int]*acc{}
	for _, rs := range rows {
		for _, r := range rs {
			a := byK[r.K]
			if a == nil {
				a = &acc{}
				byK[r.K] = a
			}
			a.n++
			a.kl += r.KL
			if r.Agree {
				a.agree++
			}
			if r.Flip {
				a.flips++
			}
		}
	}
	var ks []int
	for k := range byK {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	for _, k := range ks {
		a := byK[k]
		t.Logf("%s %s K=%d vs reference: agree %.2f%% (%d/%d), mean KL %.4f, hard flips %d", model, arm, k, 100*float64(a.agree)/float64(a.n), a.agree, a.n, a.kl/float64(a.n), a.flips)
	}
}
