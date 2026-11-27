package memprice

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

func scanBodies(t *testing.T, funcs map[string][]string, banned []string, why string) {
	t.Helper()
	for file, names := range funcs {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		f, err := parser.ParseFile(fs, file, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		found := map[string]bool{}
		bodies := map[string]ast.Node{}
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if d.Body != nil {
					bodies[d.Name.Name] = d.Body
				}
			case *ast.GenDecl:
				for _, sp := range d.Specs {
					if vs, ok := sp.(*ast.ValueSpec); ok {
						for i, n := range vs.Names {
							if i < len(vs.Values) {
								bodies[n.Name] = vs.Values[i]
							}
						}
					}
				}
			}
		}
		for _, name := range names {
			node, ok := bodies[name]
			if !ok {
				continue
			}
			found[name] = true
			body := string(src[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset])
			for _, b := range banned {
				if strings.Contains(body, b) {
					t.Errorf("%s: %s names %q: %s", file, name, b, why)
				}
			}
		}
		for _, name := range names {
			if !found[name] {
				t.Errorf("%s: no func %s to scan; the scan reads nothing for it", file, name)
			}
		}
	}
}

func TestNoFamilySwitchInPricing(t *testing.T) {
	banned := []string{"ResidentLayerOps", "Mixer", "KVSrcAtResident", "MLAResidentParams", "HeadDimAtResident", "KVHeadsAtResident",
		".Qwen35", ".LFM2", ".KDA", ".Granite", ".Nemotron", ".Gemma4", ".MLA",
		".qwen35", ".lfm2", ".kda", ".granite", ".nemotron", ".gemma4", ".mla",
		"IsLinearLayer", "IsConvLayer", "IsMambaLayer", "isLinearLayer", "isConvLayer", "isMambaLayer"}
	scanBodies(t, map[string][]string{
		"metal.go":  {"MetalKVPartsFor", "MetalRowBytesFor"},
		"cuda.go":   {"CUDAKVParts", "CUDAKVRowBytes"},
		"state.go":  {"StateSpecs", "StateBuffers", "RecurrentCheckpointBuffers"},
		"cpu.go":    {"CPUKVBytesForCtx", "CPUKVBytesForRows"},
		"webgpu.go": {"WebGPUKVBytesPerPos", "WebGPUKVRowBytes", "WebGPUKVBuffersOf"},
	}, banned, "KV and state pricing reads the operations' declarations (LayerBuffers), not the family")
}

func TestDecodeScratchNamesNoFamily(t *testing.T) {
	banned := []string{"ResidentLayerOps", "Mixer", "KVSrcAtResident", "HeadDimAtResident", "KVHeadsAtResident",
		"Qwen35", "Nemotron", "MLA", "Gemma4", "GptOss", "DeltaNet", "Mamba", "KDA", "ShortConv", "PLE", "MoEResident", "QKNormWhole",
		"NonGatedMLP", "OpAttention", "OpGatedAttn", "Ffn", ".qwen35", ".lfm2", ".kda", ".granite", ".nemotron", ".gemma4", ".mla"}
	scanBodies(t, map[string][]string{
		"metal.go": {"MetalDecodeScratchBytes", "ComputeMetalDecodeRows", "MetalRowTerms", "MetalMoETerms"},
	}, banned, "decode's scratch is priced from the operations' per-row declarations and the head's, not the family")
}

func TestMemprice(t *testing.T) {
	if got := CUDAChargedBytes(1 << 20); got != 1<<20 {
		t.Fatalf("CUDAChargedBytes(1 MiB) = %d, want %d", got, 1<<20)
	}
	if got := CUDAChargedBytes(2<<20 + 1); got != 4<<20 {
		t.Fatalf("CUDAChargedBytes(2 MiB + 1) = %d, want %d", got, 4<<20)
	}
	if got := CUDAF16Decline(darch.KVFacts{Family: "llama"}); got != "" {
		t.Fatalf("CUDAF16Decline(llama) = %q, want empty", got)
	}
	if got := CUDAF16Decline(darch.KVFacts{Family: "gpt2"}); got == "" {
		t.Fatal("CUDAF16Decline(gpt2) = empty, want decline")
	}
	if got := CUDAF16AutoDecline(darch.KVFacts{Family: "llama"}); got != "" {
		t.Fatalf("CUDAF16AutoDecline(llama) = %q, want empty", got)
	}
	if got := CUDAF16AutoDecline(darch.KVFacts{Family: "gemma3", SlidingWindow: true}); got == "" {
		t.Fatal("CUDAF16AutoDecline(sliding) = empty, want decline")
	}
	if got := CUDAI8Decline(darch.KVFacts{Family: "llama"}); got != "" {
		t.Fatalf("CUDAI8Decline(llama) = %q, want empty", got)
	}
	if got := CUDAI8Decline(darch.KVFacts{Family: "llama", KVRowNotWords: 6}); got == "" {
		t.Fatal("CUDAI8Decline(KVRowNotWords=6) = empty, want decline")
	}

	b := program.Buf{Name: "k", Unit: program.PerPosition, Prec: program.PrecKV, Elems: 64, Heads: 2}
	if got := MetalRowBytesFor(b, mem.KVF16); got != 128 {
		t.Fatalf("MetalRowBytesFor(f16) = %d, want 128", got)
	}
	if got := MetalRowBytesFor(b, mem.KVI8); got != 72 {
		t.Fatalf("MetalRowBytesFor(i8) = %d, want 72", got)
	}
	if got := CUDAKVRowBytes(b, mem.KVI8); got != 72 {
		t.Fatalf("CUDAKVRowBytes(i8) = %d, want 72", got)
	}
	if r, s := WebGPUKVRowBytes(b, false, true); r != 64 || s != 8 {
		t.Fatalf("WebGPUKVRowBytes(i8) = (%d, %d), want (64, 8)", r, s)
	}
	if got := WebGPUCtxCeilingFor(true, false); got != 32768 {
		t.Fatalf("WebGPUCtxCeilingFor(f16) = %d, want 32768", got)
	}
	if got := WebGPUVkBlocks([]int64{1024}); got != WebGPUVkDevFirst {
		t.Fatalf("WebGPUVkBlocks([1024]) = %d, want %d", got, WebGPUVkDevFirst)
	}
	if got := MetalWindowBlocksFor(512); got != 44 {
		t.Fatalf("MetalWindowBlocksFor(512) = %d, want 44", got)
	}
	if got := CPUKVBytesForCtx(nil, 1024, false, false); got != 0 {
		t.Fatalf("CPUKVBytesForCtx(nil) = %d, want 0", got)
	}
}
