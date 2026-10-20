package decoder

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The KV precision ratchet (docs/tasks/task-kv-precision-everywhere-2026-10.md, phase 0): for every backend path, family fixture and
// requested precision, ResolveKV delivers what an explicit request asks for or refuses it by name; on the CPU, which every test box has,
// each cell is also loaded and the cache it builds is checked against it. The table is docs/kv-precision-matrix.md, generated here. A
// later phase turns refusals into deliveries; a delivery that turns back into a refusal fails until the owner's word moves it.
// Regenerate: go test ./decoder -run KVPrecision -update

const kvMatrixPath = "../docs/kv-precision-matrix.md"

var kvMatrixFixtures = []struct{ group, dir string }{
	{"dense (llama)", "llama-tiny"},
	{"dense (phi3)", "phi3-tiny"},
	{"MoE", "qwen3moe-tiny"},
	{"sliding window", "gemma2-tiny"},
	{"pairwise RoPE", "cohere-tiny"},
	{"m-RoPE (VL)", "qwen25vl-tiny"},
	{"MLA", "deepseek-tiny"},
	{"Gated DeltaNet hybrid, MoE", "qwen35-tiny"},
	{"Gated DeltaNet hybrid", "qwen3next-tiny"},
	{"Mamba-2 hybrid", "nemotron-tiny"},
	{"short-conv hybrid", "lfm2-tiny"},
	{"Gemma 4, MoE", "gemma4-moe-kv-tiny"},
}

var kvMatrixValues = []string{"auto", "f32", "f16", "i8"}

type kvMatrixPathSpec struct {
	name, backend string
	set           func(*Options)
}

var kvMatrixPaths = []kvMatrixPathSpec{
	{"CUDA resident", "cuda", nil},
	{"CUDA resident with --drafter", "cuda", func(o *Options) { o.Drafter = true }},
	{"Metal resident", "metal", nil},
	{"WebGPU resident (coverage unsurveyed until phase 5)", "webgpu", nil},
	{"CPU", "cpu", nil},
	{"CPU, batched decode on", "cpu", func(o *Options) { o.CPUBatchDecode = CPUBatchOn }},
}

// kvMatrixOpts is the Options a load asking for v on p carries, as serve's flags would set them.
func kvMatrixOpts(p kvMatrixPathSpec, v string) Options {
	o := Options{Backend: p.backend}
	if p.backend == "cpu" {
		o.Backend = ""
		if v != "auto" {
			o.KVQuant = v
		}
	} else {
		o.KVPrecision = v
	}
	if p.set != nil {
		p.set(&o)
	}
	return o
}

type kvMatrixCell struct {
	delivered, reason string
	refused           bool
}

func (c kvMatrixCell) text() string {
	if c.refused {
		return "refused: " + c.reason
	}
	if c.reason != "" {
		return c.delivered + " (" + c.reason + ")"
	}
	return c.delivered
}

func kvMatrixArch(t *testing.T, dir string) *Architecture {
	t.Helper()
	cfg, err := loadConfig(os.DirFS(filepath.Join("..", "testdata", dir)), "config.json")
	if err != nil {
		t.Fatalf("%s: config: %v", dir, err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("%s: architecture: %v", dir, err)
	}
	return arch
}

// kvMatrix resolves every cell.
func kvMatrix(t *testing.T) map[[3]string]kvMatrixCell {
	out := map[[3]string]kvMatrixCell{}
	for _, f := range kvMatrixFixtures {
		arch := kvMatrixArch(t, f.dir)
		for _, p := range kvMatrixPaths {
			for _, v := range kvMatrixValues {
				o := kvMatrixOpts(p, v)
				got, why := ResolveKV(p.backend, arch, o)
				c := kvMatrixCell{delivered: got, reason: why}
				if r := kvRefusal(p.backend, arch, o); r != nil {
					c.refused, c.reason = true, r.Reason
				} else if p.backend == "cpu" && o.CPUBatchDecode == CPUBatchAuto && c.reason == "" {
					if b := cpuBatchKVBypass(got); b != "" { // delivered; batched decode under auto stands aside for it
						c.reason = "batched decode under auto: " + b
					}
				}
				out[[3]string{p.name, f.group, v}] = c
			}
		}
	}
	return out
}

func TestKVPrecisionDelivered(t *testing.T) {
	cells := kvMatrix(t)
	for k, c := range cells {
		v := k[2]
		switch {
		case kvExplicit(v) && !c.refused && c.delivered != v && c.delivered != kvPrecNone:
			t.Errorf("%s / %s / --kv %s: delivers %q without refusing", k[0], k[1], v, c.delivered)
		case c.refused && c.reason == "":
			t.Errorf("%s / %s / --kv %s: refused with no reason", k[0], k[1], v)
		case !c.refused && c.delivered == "":
			t.Errorf("%s / %s / --kv %s: delivers nothing and does not refuse", k[0], k[1], v)
		}
	}
	// The CPU cells against a real load: a refusal is Load's KVRefusal, and a delivery builds a cache of that precision.
	for _, f := range kvMatrixFixtures {
		for _, p := range kvMatrixPaths {
			if p.backend != "cpu" {
				continue
			}
			for _, v := range kvMatrixValues {
				c := cells[[3]string{p.name, f.group, v}]
				m, err := Load(filepath.Join("..", "testdata", f.dir), kvMatrixOpts(p, v))
				var r *KVRefusal
				switch {
				case c.refused:
					if !errors.As(err, &r) {
						t.Errorf("%s / %s / --kv %s: refused in the table, but Load returned %v", p.name, f.group, v, err)
					}
				case err != nil:
					t.Errorf("%s / %s / --kv %s: Load: %v", p.name, f.group, v, err)
				default:
					if got := m.CPUKVPrecision(); got != c.delivered {
						t.Errorf("%s / %s / --kv %s: CPUKVPrecision %q, the table says %q", p.name, f.group, v, got, c.delivered)
					}
					if c.delivered != kvPrecNone {
						if got := kvQuantName(m.NewCache(1).quant); got != c.delivered {
							t.Errorf("%s / %s / --kv %s: the cache is built %s, the table says %s", p.name, f.group, v, got, c.delivered)
						}
					}
				}
				if m != nil {
					m.Close()
				}
			}
		}
	}
	// The ratchet: a cell the committed table delivers must not become a refusal.
	if old, err := os.ReadFile(kvMatrixPath); err == nil {
		for k, was := range parseKVMatrix(old) {
			if c, ok := cells[k]; ok && !strings.HasPrefix(was, "refused") && c.refused {
				t.Errorf("%s / %s / --kv %s was delivered (%s) and now refuses (%s): moving a delivery back to a refusal needs the owner's word",
					k[0], k[1], k[2], was, c.reason)
			}
		}
	}
}

func kvQuantName(q kvQuant) string {
	switch q {
	case kvI8:
		return kvPrecI8
	case kvqF16:
		return kvPrecF16
	}
	return kvPrecF32
}

// TestKVPrecisionMatrix_fresh keeps docs/kv-precision-matrix.md generated from ResolveKV. Regenerate: go test ./decoder -run KVPrecision -update
func TestKVPrecisionMatrix_fresh(t *testing.T) {
	md := renderKVMatrix(kvMatrix(t))
	if *updateMatrix {
		if err := os.WriteFile(kvMatrixPath, md, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	got, err := os.ReadFile(kvMatrixPath)
	if err != nil {
		t.Fatalf("read %s: %v (regenerate: go test ./decoder -run KVPrecision -update)", kvMatrixPath, err)
	}
	if !bytes.Equal(got, md) {
		t.Fatalf("%s is out of date — regenerate: go test ./decoder -run KVPrecision -update", kvMatrixPath)
	}
}

func renderKVMatrix(cells map[[3]string]kvMatrixCell) []byte {
	var b bytes.Buffer
	w := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }
	w("# KV precision delivered, per backend path, family and request\n\n")
	w("Generated by `TestKVPrecisionMatrix_fresh` from `decoder/kvresolve.go:ResolveKV` (regenerate: `go test ./decoder -run KVPrecision -update`);\n")
	w("do not edit by hand. The contract is `docs/tasks/task-kv-precision-everywhere-2026-10.md`: an explicit `--kv f32`, `f16` or `i8` is delivered\n")
	w("or the load **refuses** and says why; `auto` is the backend's pick, with the reason when it is not the request. `TestKVPrecisionDelivered`\n")
	w("checks every cell, loads every CPU cell and checks the cache it builds, and fails a cell that moves from delivered back to refused.\n")
	w("Each family is one tiny fixture under `testdata/`; the CPU column is the CPU (host) cache, which a load uses when no resident runner is active.\n")
	for _, p := range kvMatrixPaths {
		w("\n## %s\n\n| family (fixture) |", p.name)
		for _, v := range kvMatrixValues {
			w(" `--kv %s` |", v)
		}
		w("\n|---|")
		for range kvMatrixValues {
			w("---|")
		}
		w("\n")
		for _, f := range kvMatrixFixtures {
			w("| %s (`%s`) |", f.group, f.dir)
			for _, v := range kvMatrixValues {
				w(" %s |", cells[[3]string{p.name, f.group, v}].text())
			}
			w("\n")
		}
	}
	return b.Bytes()
}

var kvMatrixRow = regexp.MustCompile(`^\| (.+) \(` + "`" + `[^` + "`" + `]+` + "`" + `\) \|(.*)\|$`)

// parseKVMatrix reads a committed matrix back into (path, family, value) -> cell text.
func parseKVMatrix(md []byte) map[[3]string]string {
	out := map[[3]string]string{}
	path := ""
	for _, ln := range strings.Split(string(md), "\n") {
		if strings.HasPrefix(ln, "## ") {
			path = strings.TrimPrefix(ln, "## ")
			continue
		}
		m := kvMatrixRow.FindStringSubmatch(ln)
		if m == nil || path == "" {
			continue
		}
		cols := strings.Split(m[2], " | ")
		for i, v := range kvMatrixValues {
			if i < len(cols) {
				out[[3]string{path, m[1], v}] = strings.TrimSpace(cols[i])
			}
		}
	}
	return out
}
