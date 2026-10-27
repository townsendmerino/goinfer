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
		wg.Add(1)
		go func() { defer wg.Done(); got[g] = answers(7, n) }()
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
