package mem

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const genGiB = int64(1) << 30

// genCurve is a KV curve: linear per position, flattened past a window when window > 0, so non-decreasing as KVCurve requires.
func genCurve(perPos int64, window int) KVCurve {
	return func(ctx int) int64 {
		if window > 0 && ctx > window {
			ctx = window
		}
		return perPos * int64(ctx)
	}
}

func genHost(r *rand.Rand) HostLoad {
	ctx := []int{0, 512, 4096, 32768, 131072}[r.IntN(5)]
	l := HostLoad{
		Name: "m", Quant: []string{"f32", "int8", "int4", "int4mix"}[r.IntN(4)],
		WeightBytes: r.Int64N(64 * genGiB), Ctx: ctx, MaxCtx: ctx, Pinned: r.IntN(3) == 0,
		NoGuard: r.IntN(20) == 0, GGUF: r.IntN(2) == 0, DenseStreamable: r.IntN(2) == 0, Int4Smaller: r.IntN(2) == 0,
	}
	if r.IntN(10) == 0 {
		l.WeightBytes = 0 // unpriced
	}
	if r.IntN(4) > 0 {
		l.KV = genCurve(r.Int64N(1<<20)+1, []int{0, 0, 4096}[r.IntN(3)])
		l.KVBytes = l.KV(ctx)
	}
	if r.IntN(2) == 0 {
		l.SourceBytes = r.Int64N(32 * genGiB)
	}
	if r.IntN(5) == 0 {
		l.ExpertBytes = r.Int64N(32 * genGiB)
		l.CUDABuildBytes = 2*l.WeightBytes + l.ExpertBytes
	}
	if r.IntN(5) == 0 {
		l.DeviceKV, l.KVBytes, l.PackBytes = true, 0, r.Int64N(genGiB)
	}
	return l
}

func genGIW(r *rand.Rand) GIWLoad {
	return GIWLoad{Ctx: []int{512, 4096, 32768, 131072}[r.IntN(4)], Pinned: r.IntN(3) == 0, KV: genCurve(r.Int64N(1<<20)+1, []int{0, 4096}[r.IntN(2)]),
		DeviceKV: r.IntN(5) == 0, Scratch: 256 << 20, PinnedExperts: []int64{0, r.Int64N(16 * genGiB)}[r.IntN(2)]}
}

func genRequest(r *rand.Rand) Request {
	return Request{KVBytes: r.Int64N(16 * genGiB), ScratchBytes: r.Int64N(genGiB), PromptTokens: r.IntN(65536), MaxTokens: r.IntN(8192), Share: r.IntN(5)}
}

func hostAnswer(f HostFit) string {
	d := "nil"
	if f.Decline != nil {
		d = fmt.Sprintf("%q %v", f.Decline.Msg, f.Decline.DenseStreamable)
	}
	return fmt.Sprintf("%d %s %q", f.PinCtx, d, f.Line)
}

func giwAnswer(f GIWFit) string {
	e := ""
	if f.Err != nil {
		e = f.Err.Error()
	}
	return fmt.Sprintf("%v %d %d %d %d %d %v %d %q %q", f.Known, f.Avail, f.Budget, f.Need, f.PinnedExperts, f.Ctx, f.Pinned, f.PinCtx, e, f.Note)
}

// answers is every decision on the i'th generated input, as text.
func answers(seed uint64, n int) []string {
	r := rand.New(rand.NewPCG(seed, 1))
	p := DefaultPolicy()
	out := make([]string, 0, n)
	for range n {
		m := HostMachine(r.Int64N(128 * genGiB))
		out = append(out, hostAnswer(SolveHost(genHost(r), m, p))+" | "+giwAnswer(SolveGIW(genGIW(r), m, p))+" | "+
			AdmitRequest(genRequest(r), m, p)+" | "+strconv.FormatInt(HostBudget(m, p), 10)+" "+strconv.FormatInt(WeightCacheBudget(r.Int64N(2)*genGiB, m), 10))
	}
	return out
}

// TestSolve_pure (MM1 gate G-MM1-S3, I4): the same inputs give the same answers, twice in one goroutine and once in each of 16
// at once, and the package keeps no state between calls (no package-level variable).
func TestSolve_pure(t *testing.T) {
	const n = 10000
	want := answers(7, n)
	if again := answers(7, n); strings.Join(again, "\n") != strings.Join(want, "\n") {
		t.Fatal("a second run in the same goroutine answered differently")
	}
	var wg sync.WaitGroup
	got := make([][]string, 16)
	for g := range got {
		wg.Go(func() { got[g] = answers(7, n) })
	}
	wg.Wait()
	for g, a := range got {
		if strings.Join(a, "\n") != strings.Join(want, "\n") {
			t.Fatalf("goroutine %d answered differently", g)
		}
	}
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	read := 0
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), e.Name(), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		read++
		for _, d := range f.Decls {
			if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.VAR {
				t.Errorf("%s declares a package-level variable: a decision could carry state from one call to the next", e.Name())
			}
		}
	}
	if read == 0 {
		t.Fatal("no source files read")
	}
}

// hostRank orders the resident guard's outcomes from worst to best: a refusal, an auto-pin, a load as asked.
func hostRank(f HostFit) int {
	switch {
	case f.Decline != nil:
		return 0
	case f.PinCtx > 0:
		return 1
	default:
		return 2
	}
}

func giwRank(f GIWFit) int {
	switch {
	case f.Err != nil:
		return 0
	case f.PinCtx > 0:
		return 1
	default:
		return 2
	}
}

// TestSolve_monotone (MM1 gate G-MM1-S4, I5): over a ladder of known readings, more available memory never gives a worse
// resident or .giw outcome or a smaller auto-pinned context, and never turns an admitted request into a refusal.
func TestSolve_monotone(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 3))
	p := DefaultPolicy()
	ladder := []int64{1, 1 << 20, 256 << 20, genGiB, 2 * genGiB, 4 * genGiB, 7 * genGiB, 12 * genGiB, 16 * genGiB, 32 * genGiB, 64 * genGiB, 128 * genGiB, 1 << 44}
	for i := range 10000 {
		hl, gl, rq := genHost(r), genGIW(r), genRequest(r)
		var ph HostFit
		var pg GIWFit
		admitted := false
		for j, a := range ladder {
			m := HostMachine(a)
			h, g, adm := SolveHost(hl, m, p), SolveGIW(gl, m, p), AdmitRequest(rq, m, p) == ""
			if j > 0 {
				if hostRank(h) < hostRank(ph) || (hostRank(h) == 1 && hostRank(ph) == 1 && h.PinCtx < ph.PinCtx) {
					t.Fatalf("input %d: the resident guard got worse from %d to %d bytes: %s -> %s", i, ladder[j-1], a, hostAnswer(ph), hostAnswer(h))
				}
				if giwRank(g) < giwRank(pg) || (giwRank(g) == 1 && giwRank(pg) == 1 && g.PinCtx < pg.PinCtx) {
					t.Fatalf("input %d: the .giw guard got worse from %d to %d bytes: %s -> %s", i, ladder[j-1], a, giwAnswer(pg), giwAnswer(g))
				}
				if admitted && !adm {
					t.Fatalf("input %d: a request admitted at %d bytes is refused at %d", i, ladder[j-1], a)
				}
			}
			ph, pg, admitted = h, g, adm
		}
	}
}

// TestImports (MM1 gate G-MM1-S2): the package imports the standard library only.
func TestImports(t *testing.T) {
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), e.Name(), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		n++
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if first, _, _ := strings.Cut(path, "/"); strings.Contains(first, ".") {
				t.Errorf("%s imports %s", e.Name(), path)
			}
		}
	}
	if n == 0 {
		t.Fatal("no source files read")
	}
}

// TestHostCheck_notOfferedStreamingWhereItCannotPage: neither a refusal nor a cap note names -stream-weights as the way out
// at a quant weight paging cannot read; at one it can read they still do.
func TestHostCheck_notOfferedStreamingWhereItCannotPage(t *testing.T) {
	const flag = "-stream-weights to lift"
	check := func(quant string) HostCheck {
		return HostCheck{Name: "m", Quant: quant, GGUF: true, MaxCtx: 4096}
	}
	for quant, pages := range map[string]bool{"q4k": false, "f32": false, "bf16": false, "f16": false,
		"int8": true, "int8int8": true, "int4": true, "int4mix": true} {
		if got := check(quant).CapNote(2048); strings.Contains(got, flag) != pages {
			t.Errorf("%s cap note: %q", quant, got)
		}
		if got := check(quant).Remedy(); strings.Contains(got, "Re-run goinfer-serve with -stream-weights") != pages {
			t.Errorf("%s remedy: %q", quant, got)
		}
		if got := QuantPages(quant); got != pages {
			t.Errorf("QuantPages(%s) = %v", quant, got)
		}
		c := check(quant)
		c.Int4Smaller = true
		// A tight load that streaming cannot help is still told what can.
		if got := c.Warning(); !pages && (!strings.Contains(got, "--quant int4") || !strings.Contains(got, "-ctx")) {
			t.Errorf("%s warning names no way out: %q", quant, got)
		}
	}
}

// TestUnpinnedExpertsHint_onlyWhenTheCopyIsWhatDoesNotFit: a refusal names --moe-experts-unpinned only when a pinned expert copy is in the need and
// dropping it would bring the need within the budget: the .giw guard's pinned-context and floor refusals, and the .gguf build's decline.
func TestUnpinnedExpertsHint_onlyWhenTheCopyIsWhatDoesNotFit(t *testing.T) {
	const GB = int64(1) << 30
	m, p := HostMachine(40*GB), DefaultPolicy()
	kv := func(ctx int) int64 { return int64(ctx) << 10 }
	for _, c := range []struct {
		name   string
		l      GIWLoad
		want   bool
		refuse bool
	}{
		{"device KV: the pinned copy is the whole need", GIWLoad{Ctx: 4096, Pinned: true, DeviceKV: true, PinnedExperts: 60 * GB, KV: kv}, true, true},
		{"host KV alone is over the budget", GIWLoad{Ctx: 50 << 20, Pinned: true, PinnedExperts: 5 * GB, KV: kv}, false, true},
		{"pinned copy only, over the budget, dropping it fits", GIWLoad{Ctx: 4096, Pinned: true, DeviceKV: true, PinnedExperts: 45 * GB, KV: kv}, true, true},
		{"floor refusal, dropping the copy fits", GIWLoad{Ctx: 4096, DeviceKV: true, PinnedExperts: 45 * GB, KV: kv}, true, true},
		{"no pinned copy", GIWLoad{Ctx: 1 << 26, Pinned: true, KV: kv}, false, true},
		{"fits", GIWLoad{Ctx: 4096, Pinned: true, DeviceKV: true, PinnedExperts: 10 * GB, KV: kv}, false, false},
	} {
		d := SolveGIW(c.l, m, p)
		if (d.Err != nil) != c.refuse {
			t.Fatalf("%s: refused %v, want %v (%v)", c.name, d.Err != nil, c.refuse, d.Err)
		}
		if d.Err == nil {
			continue
		}
		if got := strings.Contains(d.Err.Error(), "--moe-experts-unpinned"); got != c.want {
			t.Errorf("%s: the refusal names --moe-experts-unpinned: %v, want %v: %v", c.name, got, c.want, d.Err)
		}
	}
	// A .gguf CUDA expert-cache build: weights twice and the experts once.
	hc := func(w, e int64) HostCheck {
		return NewHostCheck(HostLoad{Name: "m", Quant: "int4", GGUF: true, WeightBytes: w, ExpertBytes: e, CUDABuildBytes: 2*w + e}, m, p)
	}
	if c := hc(10*GB, 12*GB); c.Fits() || !strings.Contains(c.Decline().Msg, "--moe-experts-unpinned") {
		t.Errorf("a build refused by its pinned copy (fits %v) does not name the flag: %s", c.Fits(), c.Decline().Msg)
	}
	if c := hc(20*GB, 12*GB); c.Fits() || strings.Contains(c.Decline().Msg, "--moe-experts-unpinned") {
		t.Errorf("a build that does not fit even without its copy names the flag: %s", c.Decline().Msg)
	}
	// The build peak without the copy fits, but the weights and the KV alone do not: the flag would not make the load fit, so it is not named.
	kvBound := NewHostCheck(HostLoad{Name: "m", Quant: "int4", GGUF: true, WeightBytes: 10 * GB, KVBytes: 19 * GB, ExpertBytes: 12 * GB,
		CUDABuildBytes: 32 * GB}, m, p)
	if kvBound.Fits() || strings.Contains(kvBound.Decline().Msg, "--moe-experts-unpinned") {
		t.Errorf("a load the flag would not save names it (fits %v): %s", kvBound.Fits(), kvBound.Decline().Msg)
	}
}

// TestHostCheck_heapSlackBesideTheSource: a load that maps its source is priced 1% of its weights more, for its own heap beside them
// (HeapSlackBytes); a load with no source term is not; and SmallerFittingContext takes the slack out of the budget as it does the source.
func TestHostCheck_heapSlackBesideTheSource(t *testing.T) {
	const GB = int64(1) << 30
	const w, src = 100 * GB, 50 * GB
	m, p := HostMachine(400*GB), DefaultPolicy()
	with := NewHostCheck(HostLoad{Name: "m", Quant: "int4", GGUF: true, WeightBytes: w, SourceBytes: src}, m, p)
	if got, want := with.Need(), int64(w+src+w/100); got != want {
		t.Errorf("with a source: need %d, want weights + source + 1%% of the weights = %d", got, want)
	}
	if !strings.Contains(with.Arithmetic(), "1.00 GiB heap beside it") {
		t.Errorf("the arithmetic does not name the heap slack: %q", with.Arithmetic())
	}
	without := NewHostCheck(HostLoad{Name: "m", Quant: "int4", WeightBytes: w}, m, p)
	if got := without.Need(); got != w {
		t.Errorf("with no source: need %d, want the weights alone %d", got, int64(w))
	}
	// A KV curve of 1 byte a position, with the budget leaving exactly 1,000 positions beside weights, source and slack.
	kv := func(ctx int) int64 { return int64(ctx) }
	c := HostCheck{Name: "m", GGUF: true, WeightBytes: w, SourceBytes: src, KV: kv, Ctx: 1 << 20, Policy: Policy{WeightsFraction: 1, CtxFloor: 1}}
	c.Available = w + src + w/100 + 1000
	if got, ok := c.SmallerFittingContext(); !ok || got != 1000 {
		t.Errorf("SmallerFittingContext = %d, %v; want 1000 positions once the slack is out of the budget", got, ok)
	}
}

// TestKVSlotsWithin pins the slot clamp without allocating a real clamp's worth of KV: the largest count up to the
// request whose build fits the budget, never below 1.
func TestKVSlotsWithin(t *testing.T) {
	const mb = int64(1 << 20)
	for _, tc := range []struct {
		name               string
		want               int
		budget, base, slot int64
		got                int
	}{
		{"all fit", 4, 10000 * mb, 2000 * mb, 117 * mb, 4},
		{"clamped to what fits", 8, 2500 * mb, 2000 * mb, 117 * mb, 5}, // 2000 + 4*117 = 2468 <= 2500; +5*117 > 2500
		{"exactly at the budget fits", 3, 2000*mb + 2*117*mb, 2000 * mb, 117 * mb, 3},
		{"only the first fits", 4, 2050 * mb, 2000 * mb, 117 * mb, 1},
		{"even the first over budget: still 1, the decline is elsewhere", 4, 1000 * mb, 2000 * mb, 117 * mb, 1},
		{"request of 0 means 1", 0, 10000 * mb, 2000 * mb, 117 * mb, 1},
	} {
		if got := KVSlotsWithin(tc.want, tc.budget, tc.base, tc.slot); got != tc.got {
			t.Errorf("%s: KVSlotsWithin(%d, ...) = %d, want %d", tc.name, tc.want, got, tc.got)
		}
	}
}

// TestRAMShare holds RAMShare to the arithmetic the backends' guards applied before it moved here, byte for byte: Metal's
// int64(float64(ram)*fraction) over a uint64 RAM, and WebGPU's int64(fraction*float64(ram)) over an int64 one.
func TestRAMShare(t *testing.T) {
	rams := []uint64{0, 1, 3, 1 << 30, 8 << 30, 16 << 30, 16<<30 + 7, 24 << 30, 36<<30 - 1, 64 << 30, 192 << 30, 1<<53 + 1}
	r := rand.New(rand.NewPCG(1, 2))
	for range 1000 {
		rams = append(rams, r.Uint64N(1<<40))
	}
	for _, ram := range rams {
		got := RAMShare(int64(ram))
		if metal := int64(float64(ram) * WeightsFraction); got != metal {
			t.Errorf("RAMShare(%d) = %d, Metal's uint64 arithmetic %d", ram, got, metal)
		}
		if webgpu := int64(WeightsFraction * float64(int64(ram))); got != webgpu {
			t.Errorf("RAMShare(%d) = %d, WebGPU's int64 arithmetic %d", ram, got, webgpu)
		}
	}
}

// TestWeightsPercent: the percentage the refusals print is WeightsFraction's, and "%.0f" renders it as the backends'
// own expressions did.
func TestWeightsPercent(t *testing.T) {
	if got := WeightsPercent(); got != WeightsFraction*100 {
		t.Errorf("WeightsPercent() = %v, want WeightsFraction*100 = %v", got, WeightsFraction*100)
	}
	if got, want := fmt.Sprintf("%.0f%%", WeightsPercent()), fmt.Sprintf("%.0f%%", 100*WeightsFraction); got != want {
		t.Errorf("WeightsPercent prints %q, the backends printed %q", got, want)
	}
}

// TestSnapshotHostBudget pins the default prefix-snapshot host budget (-kv-host-bytes auto): 1/KVHostFraction of physical
// RAM up to KVHostCap, 0 when physical RAM is unknown, and formatted with GiB.
func TestSnapshotHostBudget(t *testing.T) {
	const g = int64(1) << 30
	p := DefaultPolicy()
	for _, tc := range []struct {
		ram  int64
		want int64
		how  string
	}{
		{0, 0, "unknown"},
		{-1, 0, "unknown"},
		{16 * g, 16 * g / 10, "auto: 1/10 of 16.00 GiB of RAM, at most 8.00 GiB"},
		{64 * g, 64 * g / 10, "auto: 1/10 of 64.00 GiB of RAM, at most 8.00 GiB"},
		{128 * g, 8 * g, "auto: 1/10 of 128.00 GiB of RAM, at most 8.00 GiB"},
	} {
		got, how := SnapshotHostBudget(tc.ram, p)
		if got != tc.want || !strings.Contains(how, tc.how) {
			t.Errorf("SnapshotHostBudget(%d) = (%d, %q); want (%d, containing %q)", tc.ram, got, how, tc.want, tc.how)
		}
	}
}
