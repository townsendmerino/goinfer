//go:build darwin

package metal

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalResidentCheckCap gates C3 (Metal half): writes at/past the resident KV cap are refused
// — a real device write there is out-of-bounds and, on unified memory, silently corrupts adjacent
// MTLBuffers. Pure logic (checkCap only reads ctxCap()), so no Metal device is needed. A
// zero-value &metalResident{} (r == nil) deliberately exercises ctxCap()'s nil-safe fallback to
// metalCtxCapDefault (G6, docs/tasks/task-gpu-paths-2026-09.md added resident.ctxCap as a per-build,
// request-aware value; this test predates that and is meant to keep testing "the historical
// ceiling" as pure logic, not require a real *resident).
func TestMetalResidentCheckCap(t *testing.T) {
	r := &metalResident{}
	if r.ContextCap() != metalCtxCapDefault {
		t.Fatalf("ContextCap = %d, want %d", r.ContextCap(), metalCtxCapDefault)
	}
	for _, c := range []struct {
		pos, n int
		ok     bool
	}{
		{0, 1, true}, {metalCtxCapDefault - 1, 1, true}, {metalCtxCapDefault, 1, false},
		{0, metalCtxCapDefault, true}, {0, metalCtxCapDefault + 1, false}, {-1, 1, false},
	} {
		err := r.checkCap(c.pos, c.n)
		if (err == nil) != c.ok {
			t.Errorf("checkCap(%d,%d) err=%v, want ok=%v", c.pos, c.n, err, c.ok)
		}
	}
}

// TestMetalCtxCapWithinKernelBound pins the invariant that keeps the resident context ceiling a
// FACT: checkCap bounds nKeys to ctxCap() <= metalCtxCapMax (32768). The attention kernel operates
// with a static threadgroup score tile buffer `threadgroup float sc[4096]` (attnScoreTileBound).
// Deep context (>4096) is handled via online softmax tiling in multiples of attnScoreTileBound.
// This test asserts metalCtxCapDefault <= attnScoreTileBound and metalCtxCapMax is a multiple of attnScoreTileBound.
func TestMetalCtxCapWithinKernelBound(t *testing.T) {
	if metalCtxCapDefault > attnScoreTileBound {
		t.Fatalf("metalCtxCapDefault=%d exceeds attention kernel tile bound %d",
			metalCtxCapDefault, attnScoreTileBound)
	}
	if metalCtxCapMax%attnScoreTileBound != 0 {
		t.Fatalf("metalCtxCapMax=%d is not a multiple of attention tile bound %d",
			metalCtxCapMax, attnScoreTileBound)
	}
}

// TestResolveMetalCtxCap is G6's own gate for the real, pre-existing gap found scoping it
// (docs/tasks/task-gpu-paths-2026-09.md): Metal never read decoder.Model.ResidentContextRequest() at
// all, so an explicit -ctx was silently ignored, always using metalCtxCapDefault. Uses
// testdata/llama-tiny (TRACKED in git, max_position_embeddings=128), so every case runs in CI
// unconditionally — no device needed, resolveMetalCtxCap is pure logic over decoder.Model state.
func TestResolveMetalCtxCap(t *testing.T) {
	load := func(t *testing.T, ctxReq int) *decoder.Model {
		t.Helper()
		m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "f32", ResidentContext: ctxReq})
		if err != nil {
			t.Fatalf("Load(ResidentContext=%d): %v", ctxReq, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}

	t.Run("unset ⇒ metalCtxCapDefault, unchanged historical default", func(t *testing.T) {
		m := load(t, 0)
		got, err := resolveMetalCtxCap(m)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != metalCtxCapDefault {
			t.Errorf("resolveMetalCtxCap = %d, want %d", got, metalCtxCapDefault)
		}
	})

	t.Run("explicit, fits both the kernel ceiling and the model window ⇒ honoured exactly", func(t *testing.T) {
		m := load(t, 64) // llama-tiny's window is 128; 64 < both ceilings
		got, err := resolveMetalCtxCap(m)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != 64 {
			t.Errorf("resolveMetalCtxCap = %d, want 64 (the explicit request, NOT metalCtxCapDefault — "+
				"this is the bug: it used to always return %d regardless)", got, metalCtxCapDefault)
		}
	})

	t.Run("explicit, exceeds the model's own window ⇒ clamped to the window, not refused", func(t *testing.T) {
		m := load(t, 200) // > llama-tiny's 128-position window, but well under metalCtxCapMax
		got, err := resolveMetalCtxCap(m)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != 128 {
			t.Errorf("resolveMetalCtxCap = %d, want 128 (clamped to the model's own window, same as CUDA's resolveCtxCap)", got)
		}
	})

	t.Run("explicit, exceeds the kernel's hard ceiling ⇒ REFUSED with the numbers, never silently clamped", func(t *testing.T) {
		m := load(t, metalCtxCapMax+1)
		_, err := resolveMetalCtxCap(m)
		if err == nil {
			t.Fatal("resolveMetalCtxCap succeeded for a request above metalCtxCapMax — should refuse")
		}
		for _, want := range []string{fmt.Sprintf("%d", metalCtxCapMax+1), fmt.Sprintf("%d", metalCtxCapMax)} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not mention %q — a refusal must name the numbers", err, want)
			}
		}
	})
}

// TestMetalBuildResident_explicitCtxTooLargeRefusesNotDecline confirms the wiring one level up:
// the wrapper (metalBackend.BuildResident) must surface resolveMetalCtxCap's error as a REAL,
// named err — not swallow it into the generic ok=false/err=nil decline every other unsupported
// shape gets — mirroring CUDA's own errKVWontFit precedent (an operator's explicit request that
// cannot be honoured gets a specific, propagated reason, not an opaque "declined").
func TestMetalBuildResident_explicitCtxTooLargeRefusesNotDecline(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int4", ResidentContext: metalCtxCapMax + 1})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	b := &metalBackend{}
	_, ok, err := b.BuildResident(m)
	if ok {
		t.Fatal("BuildResident succeeded for an explicit ctx above metalCtxCapMax — should refuse")
	}
	if err == nil {
		t.Fatal("BuildResident returned ok=false, err=nil — should be a NAMED error (G6), not a generic decline")
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("%d", metalCtxCapMax)) {
		t.Errorf("error %q does not name metalCtxCapMax=%d", err, metalCtxCapMax)
	}
}

// TestMetalBuildResident_explicitCtxHonoured is the positive twin: a SMALLER explicit -ctx (well
// within both ceilings) must actually build a working resident whose ContextCap() reflects the
// request — not the historical metalCtxCapMax — and that resident must still decode correctly at
// that smaller capacity. Before this fix, an explicit -ctx was silently ignored entirely; this
// pins that the fix's honored path is not just accepted (BuildResident succeeds) but genuinely
// EFFECTIVE (ContextCap changed, a real forward pass still runs).
func TestMetalBuildResident_explicitCtxHonoured(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const want = 32 // well under both llama-tiny's 128-window and metalCtxCapMax
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int4", ResidentContext: want})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	b := &metalBackend{}
	rf, ok, err := b.BuildResident(m)
	if err != nil || !ok {
		t.Fatalf("BuildResident(ResidentContext=%d): ok=%v err=%v", want, ok, err)
	}
	defer b.Close()
	capped, isCapped := rf.(decoder.ResidentCapped)
	if !isCapped {
		t.Fatalf("resident does not implement decoder.ResidentCapped: %T", rf)
	}
	if got := capped.ContextCap(); got != want {
		t.Fatalf("ContextCap() = %d, want %d — the explicit request did not reach the built resident", got, want)
	}
	// A real forward pass at the smaller cap must still produce finite, sane logits — not just
	// "ContextCap reports the right number while decode itself is broken".
	emb := make([]float32, m.Config().HiddenDim)
	emb[0] = 1
	logits, ferr := rf.Forward(emb, 0)
	if ferr != nil {
		t.Fatalf("Forward at pos 0 with ContextCap=%d: %v", want, ferr)
	}
	if len(logits) != m.Config().VocabSize {
		t.Fatalf("logits len = %d, want vocab %d", len(logits), m.Config().VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits[%d] = %v — degenerate output at the smaller resident context", i, v)
		}
	}
}
