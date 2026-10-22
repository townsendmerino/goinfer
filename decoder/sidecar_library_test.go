package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// MEM1-G, Load's sidecar default for a library caller (docs/tasks/task-memory-campaign-2026-10.md, §4 MEM1, registered
// before the code): G0 the default in a non-test program, G1 what it opens, G2 the opt-out, G3 the test-binary default,
// G4 the refusal.

func kvStrArr(k string, vs []string) ggufKV {
	b := append(gU32(gtString), gU64(uint64(len(vs)))...)
	for _, v := range vs {
		b = append(b, gStr(v)...)
	}
	return ggufKV{k, gtArray, b}
}

// tokenizerTinyGGUF writes a one-layer llama .gguf that carries a tokenizer into dir and returns its path. The sidecar
// transcode refuses a .gguf with no tokenizer, and no committed tiny .gguf has one.
func tokenizerTinyGGUF(t *testing.T, dir string) string {
	t.Helper()
	const arch, hidden, heads, kvHeads, hd, inter, vocab = "llama", 32, 4, 2, 8, 64, 64
	qDim, kvDim := heads*hd, kvHeads*hd
	toks := make([]string, vocab)
	for i := range toks {
		toks[i] = string(rune('a'+i%26)) + strings.Repeat("x", i/26)
	}
	kvs := []ggufKV{
		kvStr("general.architecture", arch),
		kvU32(arch+".context_length", 64), kvU32(arch+".embedding_length", hidden),
		kvU32(arch+".block_count", 1), kvU32(arch+".attention.head_count", heads),
		kvU32(arch+".attention.head_count_kv", kvHeads), kvU32(arch+".feed_forward_length", inter),
		kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6), kvF32(arch+".rope.freq_base", 10000),
		kvStr("tokenizer.ggml.model", "gpt2"), kvStrArr("tokenizer.ggml.tokens", toks),
	}
	wave := func(n, seed int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(0.1 * math.Sin(float64(seed*7919+i)))
		}
		return v
	}
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	ts := []ggufDataTensor{
		{"token_embd.weight", []uint64{hidden, vocab}, wave(vocab*hidden, 1)},
		{"output_norm.weight", []uint64{hidden}, ones(hidden)},
		{"output.weight", []uint64{hidden, vocab}, wave(vocab*hidden, 2)},
		{"blk.0.attn_norm.weight", []uint64{hidden}, ones(hidden)},
		{"blk.0.attn_q.weight", []uint64{hidden, uint64(qDim)}, wave(qDim*hidden, 3)},
		{"blk.0.attn_k.weight", []uint64{hidden, uint64(kvDim)}, wave(kvDim*hidden, 4)},
		{"blk.0.attn_v.weight", []uint64{hidden, uint64(kvDim)}, wave(kvDim*hidden, 5)},
		{"blk.0.attn_output.weight", []uint64{uint64(qDim), hidden}, wave(hidden*qDim, 6)},
		{"blk.0.ffn_norm.weight", []uint64{hidden}, ones(hidden)},
		{"blk.0.ffn_gate.weight", []uint64{hidden, inter}, wave(inter*hidden, 7)},
		{"blk.0.ffn_up.weight", []uint64{hidden, inter}, wave(inter*hidden, 8)},
		{"blk.0.ffn_down.weight", []uint64{inter, hidden}, wave(hidden*inter, 9)},
	}
	p := filepath.Join(dir, "tok-tiny.gguf")
	if err := os.WriteFile(p, buildGGUFData(kvs, ts), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// sidecarDefaultOn turns the library default on inside this test binary.
func sidecarDefaultOn(t *testing.T) {
	sidecarInTests = true
	t.Cleanup(func() { sidecarInTests = false })
}

const mem1Quant = "int8int8"

var mem1Prompt = []int{3, 14, 15, 9, 26, 5, 35}

// mem1Logits is the logits after each prompt token, concatenated.
func mem1Logits(t *testing.T, m *Model) []float32 {
	t.Helper()
	cache := m.NewCache(len(mem1Prompt))
	var all []float32
	for _, id := range mem1Prompt {
		lg, _, err := m.ForwardCapture(id, cache, nil)
		if err != nil {
			t.Fatal(err)
		}
		all = append(all, lg...)
	}
	return all
}

func dirNames(t *testing.T, dir string) []string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range ents {
		names = append(names, e.Name())
	}
	return names
}

func mem1NeedsSidecarPlatform(t *testing.T) {
	if !DefaultToSidecar(false) {
		t.Skipf("the sidecar default is darwin and linux only; this is %s", runtime.GOOS)
	}
}

// G1: the library default opens the sidecar the app path builds, builds it once, and computes what the app path's two
// steps (EnsureCachedGIW, then Load of the .giw) compute.
func TestLoad_libraryDefaultOpensTheSidecar(t *testing.T) {
	mem1NeedsSidecarPlatform(t)
	src := tokenizerTinyGGUF(t, t.TempDir())
	sidecarDefaultOn(t)

	m, err := Load(src, Options{Quant: mem1Quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	want := SidecarPath(src, mem1Quant, "cpu", false)
	if m.srcPath != want || m.mmap == nil {
		t.Fatalf("Load opened %q (mapped %v), want the sidecar %q, mapped", m.srcPath, m.mmap != nil, want)
	}
	fi1, err := os.Stat(want)
	if err != nil {
		t.Fatal(err)
	}

	again, err := Load(src, Options{Quant: mem1Quant})
	if err != nil {
		t.Fatalf("second Load: %v", err)
	}
	defer again.Close()
	if fi2, err := os.Stat(want); err != nil || !fi2.ModTime().Equal(fi1.ModTime()) {
		t.Fatalf("the second Load rewrote the sidecar (mtime %v, then %v, err %v)", fi1.ModTime(), fi2.ModTime(), err)
	}

	p, err := EnsureCachedGIW(context.Background(), src, mem1Quant, "cpu", false)
	if err != nil || p != want {
		t.Fatalf("EnsureCachedGIW = %q, %v; want %q", p, err, want)
	}
	app, err := Load(p, Options{Quant: mem1Quant})
	if err != nil {
		t.Fatalf("Load(%s): %v", p, err)
	}
	defer app.Close()
	if got, ref := mem1Logits(t, m), mem1Logits(t, app); !slices.Equal(got, ref) {
		t.Fatal("the library default's logits differ from the app path's (EnsureCachedGIW, then Load of the .giw)")
	}
}

// G2: DirectLoad loads the source into the heap and writes nothing beside it.
func TestLoad_directLoadWritesNothing(t *testing.T) {
	mem1NeedsSidecarPlatform(t)
	dir := t.TempDir()
	src := tokenizerTinyGGUF(t, dir)
	sidecarDefaultOn(t)
	m, err := Load(src, Options{Quant: mem1Quant, DirectLoad: true})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.mmap != nil || m.srcPath == SidecarPath(src, mem1Quant, "cpu", false) {
		t.Fatalf("DirectLoad opened %q (mapped %v), want the heap load of the source", m.srcPath, m.mmap != nil)
	}
	if got := dirNames(t, dir); !slices.Equal(got, []string{filepath.Base(src)}) {
		t.Fatalf("DirectLoad wrote beside the source: %v", got)
	}
}

// G3: in a test binary the zero Options keeps the direct load, so a test that loads a fixture writes nothing beside it.
func TestLoad_testBinaryDefaultIsDirect(t *testing.T) {
	dir := t.TempDir()
	src := tokenizerTinyGGUF(t, dir)
	m, err := Load(src, Options{Quant: mem1Quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.mmap != nil {
		t.Fatal("a test binary's zero Options opened a mapping, want the direct load")
	}
	if got := dirNames(t, dir); !slices.Equal(got, []string{filepath.Base(src)}) {
		t.Fatalf("a test binary's load wrote beside the source: %v", got)
	}
}

// G4: a .gguf whose sidecar cannot be written is refused, naming the opt-out, and nothing is left beside it.
func TestLoad_sidecarRefusalNamesDirectLoad(t *testing.T) {
	mem1NeedsSidecarPlatform(t)
	if os.Geteuid() == 0 {
		t.Skip("root writes into a read-only directory, so there is no refusal to see")
	}
	dir := t.TempDir()
	src := tokenizerTinyGGUF(t, dir)
	if err := os.Chmod(dir, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o755) })
	sidecarDefaultOn(t)
	m, err := Load(src, Options{Quant: mem1Quant})
	if err == nil {
		m.Close()
		t.Fatal("Load succeeded in a read-only directory; want the sidecar's refusal")
	}
	if !strings.Contains(err.Error(), "Options.DirectLoad") {
		t.Fatalf("the refusal does not name Options.DirectLoad: %v", err)
	}
	if got := dirNames(t, dir); !slices.Equal(got, []string{filepath.Base(src)}) {
		t.Fatalf("a refused load left files beside the source: %v", got)
	}
}

// G0: a non-test program's Options, with only a Quant set, takes the sidecar with no hook: testdata/mem1-libload loads the
// .gguf as a library caller does and prints its greedy tokens, which must equal this process's load of the sidecar.
func TestLoad_productionDefaultTakesTheSidecar(t *testing.T) {
	mem1NeedsSidecarPlatform(t)
	if testing.Short() {
		t.Skip("builds and runs a program")
	}
	src := tokenizerTinyGGUF(t, t.TempDir())
	out, err := exec.Command("go", "run", "./testdata/mem1-libload", src, mem1Quant).CombinedOutput()
	if err != nil {
		t.Fatalf("go run ./testdata/mem1-libload: %v\n%s", err, out)
	}
	want := SidecarPath(src, mem1Quant, "cpu", false)
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("the program's load wrote no sidecar at %s: %v\n%s", want, err, out)
	}
	m, err := Load(want, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ref := greedyTokens(t, m, mem1Prompt, 8)
	var line string
	for l := range strings.Lines(string(out)) {
		if after, ok := strings.CutPrefix(l, "tokens "); ok {
			line = strings.TrimSpace(after)
		}
	}
	if line != fmt.Sprint(ref) {
		t.Fatalf("the program's tokens %q differ from the sidecar's %v\n%s", line, ref, out)
	}
}
