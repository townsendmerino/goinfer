# Forward-architecture census, refreshed against program-based admission (RF0 close-out, 2026-11-07)

**Status: exploratory, read from code and run once.** It closes the census half of RF0
(`docs/tasks/task-forward-architecture-2026-10.md` §5 and §7a). It replaces §2 of
`docs/measurements/forward-arch-baseline-2026-10-13.md`, which was taken at main `9593bd70` before admission read operation
tables. Taken at main `912eae66` on the MacBook (Apple M1 Pro, 16 GB, go1.27.0 darwin/arm64), by day, under the shared timing
lock. Nothing was timed; no number here is a speed, and nothing in it is a gate.

## Result

**Which families decode on the CPU, per GPU backend** (the registry has 52 keys and 43 distinct `Architecture.Name`s, as on 10-13):

| basis | CUDA | Metal | WebGPU |
|---|---|---|---|
| 2026-10-13, `ResidentEligible` (the baseline's own count) | 10 | 12 | 16 |
| now, `ResidentEligible` = runner shape and capability gates (what `docs/hardware-matrix.md` prints) | **8** | **8** | **13** |
| now, `Admit(...).Why` in the matrix's model-free configuration (adds the precision policy) | 9 | 9 | 14 |
| now, `Admit(...).Why` in the configuration the loader picks (CUDA per-32 activation scales; Metal `--quant q4k`) | 8 | 8 | 14 |

- **Left the CPU-only list since 10-13** (by `ResidentEligible`; the counts in this record are that row unless a row says `Admit(...).Why`):
  - Gemma 2 on CUDA and Metal (softcap declared: `b81635ef` for CUDA; Metal's declaration, OP1-M1).
  - DeepSeek-V2 and V3 on Metal (MLA, OP2a, `f784fbdd`).
  - Olmo 3, SmolLM3 and Olmo Hybrid on WebGPU (OW, `af351b21`).
  - Nemotron-H on CUDA and Metal is **not a lift**: its int4-only precision policy moved out of the capability gate into `Admit`'s `Why`, where it applies on all three backends. So each `Admit(...).Why` row is the `ResidentEligible` row plus one (8 / 8 / 13 becomes 9 / 9 / 14 model-free); a default int8 Nemotron-H load decodes on the CPU on CUDA, Metal and WebGPU alike unless it loads at int4 or `GOINFER_SSM_RESIDENT` is set. Nemotron-H itself is resident at int4 on all three (`56378ef5` and `c6422a4b` for CUDA and Metal).
- **Entered the list:** none.
- **Still CPU-only on all three** (these 7 were on all three lists on 10-13 too; Ling 3.0 was called `bailing_hybrid` there): Granite-4.0-H (bring-up gate), LFM2.5 (own forward, not bridged), Laguna and Spark-X2.5 (local and global RoPE tables differ in length), Llama 4 (own forward, not bridged), Ling 3.0 (`kda`), and Phi-3/Phi-4 under the matrix's model-free configuration. GPT-2 is CPU-only on CUDA and WebGPU (learned position embeddings, non-gated GELU-tanh); Kimi K2 on Metal only (384 experts against a router of 256).
- **Phi-3/Phi-4 is the one family where the matrix and a load differ:** the activation hazard clears under CUDA's per-32 activation scales and Metal's `--quant q4k`, so a default load there is resident (the output below prints both reasons, the model-free one and the loader's pick, which is empty for it); on WebGPU it stays on the CPU. That is why the loader's-pick row is 8 / 8 / 14: Nemotron-H's policy adds one to each, and Phi-3 comes off CUDA and Metal.
- **WebGPU's 13** (`ResidentEligible`) are six declines for missing features, not shapes (Command-R, Command-R7B, GLM-OCR, GPT-2, Gemma 2, Ministral 3 for `attn-temp`) and the 7 families listed as CPU-only on all three above. Under `Admit(...).Why` it is 14, the extra one being Nemotron-H's precision policy; the loader's pick does not change it, since Phi-3 stays on the CPU there.

**What changed in how a decline is stated:**
- CUDA and Metal admission now reads an operation table (`program.CUDA`, `program.Metal`: OP4cu-c2, OP4m-c2), so a gap names the layer and the operation (`layer 0: operation 1 (kda): cuda has no step for it`), and every gap is listed, not the first.
- WebGPU's admission still reads a feature set (`webgpu does not implement [...]`): the census prints `admission reads a program table: false` for it. OP4 on WebGPU is the open edge.
- The runner-shape check still comes first and is independent of the tables.

**The size of the seam RF1 starts from, unchanged:** `decoder` declares 49 `Resident*` interfaces (`type Resident... interface`), as on 10-13, and `decoder/residency.go` is 1,976 lines (1,975 then).

## What ran

A temporary test in package `decoder`, not committed, the baseline's walk with `Admit` in place of `ResidentEligible` and `residentGateReason`:
- it walks the registry, resolves each key's `representativeConfig`, dedupes by `Architecture.Name`;
- for each backend it calls `Admit(archAdmitFacts(arch, backend))`, and again with the loader's pick (`ActGroup = 32` on CUDA, `Quant = "q4k"` on Metal);
- it prints, for each cell that declines in either configuration, the runner shape, the gate count and both reasons (the model-free one and the loader's pick), and the counts.

`GOINFER_SSM_RESIDENT` and `GOINFER_GEMMA4_RESIDENT` are empty, as the matrix generator sets them. Run from a worktree at `912eae66`:
`go test ./decoder -run '^TestZZCensus2$' -count=1 -v`.

```
backend cuda: admission reads a program table: true
backend metal: admission reads a program table: true
backend webgpu: admission reads a program table: false
bailing_hybrid   cuda    default+loader                           shape="" gates=1
    model-free why="layer 0: operation 1 (kda): cuda has no step for it"
    loader's-pick why="layer 0: operation 1 (kda): cuda has no step for it" (keys: bailing_hybrid)
bailing_hybrid   metal   default+loader                           shape="" gates=1
    model-free why="layer 0: operation 1 (kda): metal has no step for it"
    loader's-pick why="layer 0: operation 1 (kda): metal has no step for it" (keys: bailing_hybrid)
bailing_hybrid   webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [kda], which this model needs"
    loader's-pick why="webgpu does not implement [kda], which this model needs" (keys: bailing_hybrid)
cohere           webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [layer-norm pairwise-rope parallel-block], which this model needs"
    loader's-pick why="webgpu does not implement [layer-norm pairwise-rope parallel-block], which this model needs" (keys: cohere)
cohere2          webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [layer-norm pairwise-rope parallel-block], which this model needs"
    loader's-pick why="webgpu does not implement [layer-norm pairwise-rope parallel-block], which this model needs" (keys: cohere2)
gemma2           webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [attn-logit-softcap], which this model needs"
    loader's-pick why="webgpu does not implement [attn-logit-softcap], which this model needs" (keys: gemma2)
glm_ocr          webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [pairwise-mrope pairwise-rope], which this model needs"
    loader's-pick why="webgpu does not implement [pairwise-mrope pairwise-rope], which this model needs" (keys: glm_ocr, glm_ocr_text)
gpt2             cuda    default+loader                           shape="" gates=1
    model-free why="the embedding stage: cuda does not implement learned position embeddings; layer 0: operation 6 (ffn): cuda has no GELU-tanh activation in its non-gated block"
    loader's-pick why="the embedding stage: cuda does not implement learned position embeddings; layer 0: operation 6 (ffn): cuda has no GELU-tanh activation in its non-gated block" (keys: gpt2)
gpt2             webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [layer-norm learned-pos non-gated-gelu-tanh], which this model needs"
    loader's-pick why="webgpu does not implement [layer-norm learned-pos non-gated-gelu-tanh], which this model needs" (keys: gpt2)
granitemoehybrid cuda    default+loader                           shape="bring-up gate (GOINFER_SSM_RESIDENT unset)" gates=1
    model-free why="arch is not eligible for the resident decode runner: bring-up gate (GOINFER_SSM_RESIDENT unset)"
    loader's-pick why="arch is not eligible for the resident decode runner: bring-up gate (GOINFER_SSM_RESIDENT unset)" (keys: granitemoehybrid)
granitemoehybrid metal   default+loader                           shape="bring-up gate (GOINFER_SSM_RESIDENT unset)" gates=1
    model-free why="arch is not eligible for the resident decode runner: bring-up gate (GOINFER_SSM_RESIDENT unset)"
    loader's-pick why="arch is not eligible for the resident decode runner: bring-up gate (GOINFER_SSM_RESIDENT unset)" (keys: granitemoehybrid)
granitemoehybrid webgpu  default+loader                           shape="bring-up gate (GOINFER_SSM_RESIDENT unset)" gates=0
    model-free why="arch is not eligible for the resident decode runner: bring-up gate (GOINFER_SSM_RESIDENT unset)"
    loader's-pick why="arch is not eligible for the resident decode runner: bring-up gate (GOINFER_SSM_RESIDENT unset)" (keys: granitemoehybrid)
kimi_k2          metal   default+loader                           shape="" gates=1
    model-free why="MoE with 384 experts / 1 expert groups exceeds metal's router capacity (256 / 64)"
    loader's-pick why="MoE with 384 experts / 1 expert groups exceeds metal's router capacity (256 / 64)" (keys: kimi_k2)
laguna           cuda    default+loader                           shape="local and global RoPE tables differ in length" gates=2
    model-free why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length"
    loader's-pick why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length" (keys: laguna)
laguna           metal   default+loader                           shape="local and global RoPE tables differ in length" gates=2
    model-free why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length"
    loader's-pick why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length" (keys: laguna)
laguna           webgpu  default+loader                           shape="local and global RoPE tables differ in length" gates=1
    model-free why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length"
    loader's-pick why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length" (keys: laguna)
lfm2             cuda    default+loader                           shape="own forward, not bridged" gates=1
    model-free why="arch is not eligible for the resident decode runner: own forward, not bridged"
    loader's-pick why="arch is not eligible for the resident decode runner: own forward, not bridged" (keys: lfm2, lfm2_vl)
lfm2             metal   default+loader                           shape="own forward, not bridged" gates=1
    model-free why="arch is not eligible for the resident decode runner: own forward, not bridged"
    loader's-pick why="arch is not eligible for the resident decode runner: own forward, not bridged" (keys: lfm2, lfm2_vl)
lfm2             webgpu  default+loader                           shape="own forward, not bridged" gates=1
    model-free why="arch is not eligible for the resident decode runner: own forward, not bridged"
    loader's-pick why="arch is not eligible for the resident decode runner: own forward, not bridged" (keys: lfm2, lfm2_vl)
llama4_text      cuda    default+loader                           shape="own forward, not bridged" gates=1
    model-free why="arch is not eligible for the resident decode runner: own forward, not bridged"
    loader's-pick why="arch is not eligible for the resident decode runner: own forward, not bridged" (keys: llama4_text)
llama4_text      metal   default+loader                           shape="own forward, not bridged" gates=1
    model-free why="arch is not eligible for the resident decode runner: own forward, not bridged"
    loader's-pick why="arch is not eligible for the resident decode runner: own forward, not bridged" (keys: llama4_text)
llama4_text      webgpu  default+loader                           shape="own forward, not bridged" gates=0
    model-free why="arch is not eligible for the resident decode runner: own forward, not bridged"
    loader's-pick why="arch is not eligible for the resident decode runner: own forward, not bridged" (keys: llama4_text)
mistral3         webgpu  default+loader                           shape="" gates=1
    model-free why="webgpu does not implement [attn-temp], which this model needs"
    loader's-pick why="webgpu does not implement [attn-temp], which this model needs" (keys: ministral3, mistral3)
nemotron_h       cuda    default+loader                           shape="" gates=0
    model-free why="Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)"
    loader's-pick why="Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)" (keys: nemotron_h)
nemotron_h       metal   default+loader                           shape="" gates=0
    model-free why="Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)"
    loader's-pick why="Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)" (keys: nemotron_h)
nemotron_h       webgpu  default+loader                           shape="" gates=0
    model-free why="Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)"
    loader's-pick why="Nemotron-H runs resident with int4 projections only by default (load with Quant int4, or set GOINFER_SSM_RESIDENT to allow int8)" (keys: nemotron_h)
phi3             cuda    default only (the loader's pick admits)  shape="" gates=1
    model-free why="every resident cuda projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)"
    loader's-pick why="" (keys: phi3)
phi3             metal   default only (the loader's pick admits)  shape="" gates=1
    model-free why="every resident metal projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)"
    loader's-pick why="" (keys: phi3)
phi3             webgpu  default+loader                           shape="" gates=1
    model-free why="every resident webgpu projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)"
    loader's-pick why="every resident webgpu projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)" (keys: phi3)
spark2_5         cuda    default+loader                           shape="local and global RoPE tables differ in length" gates=1
    model-free why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length"
    loader's-pick why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length" (keys: spark2_5)
spark2_5         metal   default+loader                           shape="local and global RoPE tables differ in length" gates=1
    model-free why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length"
    loader's-pick why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length" (keys: spark2_5)
spark2_5         webgpu  default+loader                           shape="local and global RoPE tables differ in length" gates=1
    model-free why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length"
    loader's-pick why="arch is not eligible for the resident decode runner: local and global RoPE tables differ in length" (keys: spark2_5)
registry keys 52; distinct Architecture.Name 43; CPU-only by default: cuda 9, metal 9, webgpu 14; under the loader's pick: cuda 8, metal 8, webgpu 14
```

The test file:

```go
package decoder

// Temporary (not committed): the RF0 census refresh. Every registry family on each GPU backend through Admit, in the matrix's
// model-free configuration and in the configuration the loader picks for an activation-hazard family (cuda: per-32 activation
// scales; metal: --quant q4k).

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
)

func TestZZCensus2(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	seen := map[string][]string{}
	var order []string
	archs := map[string]*Architecture{}
	for _, mt := range keys {
		cfg := representativeConfig(mt)
		if cfg == nil {
			fmt.Printf("SKIP %s: no representative config\n", mt)
			continue
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			fmt.Printf("SKIP %s: resolve: %v\n", mt, err)
			continue
		}
		if _, ok := seen[arch.Name]; !ok {
			order = append(order, arch.Name)
			archs[arch.Name] = arch
		}
		seen[arch.Name] = append(seen[arch.Name], mt)
	}
	var b strings.Builder
	cpuDefault, cpuLoader := map[string]int{}, map[string]int{}
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		fmt.Fprintf(&b, "backend %s: admission reads a program table: %v\n", be, residentBackendTables[be] != nil)
	}
	for _, name := range order {
		arch := archs[name]
		for _, be := range []string{"cuda", "metal", "webgpu"} {
			def := Admit(archAdmitFacts(arch, be))
			lf := archAdmitFacts(arch, be)
			switch be {
			case "cuda":
				lf.ActGroup = 32
			case "metal":
				lf.Quant = "q4k"
			}
			ld := Admit(lf)
			if def.Why != "" {
				cpuDefault[be]++
			}
			if ld.Why != "" {
				cpuLoader[be]++
			}
			if def.Why == "" && ld.Why == "" {
				continue
			}
			tag := "default+loader"
			if def.Why != "" && ld.Why == "" {
				tag = "default only (the loader's pick admits)"
			}
			fmt.Fprintf(&b, "%-16s %-7s %-40s shape=%q gates=%d\n    model-free why=%q\n    loader's-pick why=%q (keys: %s)\n", name, be, tag, def.Shape, len(def.Gates), def.Why, ld.Why, strings.Join(seen[name], ", "))
		}
	}
	fmt.Fprintf(&b, "registry keys %d; distinct Architecture.Name %d; CPU-only by default: cuda %d, metal %d, webgpu %d; under the loader's pick: cuda %d, metal %d, webgpu %d\n",
		len(keys), len(order), cpuDefault["cuda"], cpuDefault["metal"], cpuDefault["webgpu"], cpuLoader["cuda"], cpuLoader["metal"], cpuLoader["webgpu"])
	os.Stdout.WriteString(b.String())
}
```

## What RF0 still has open

- **Done:** the per-turn `timings` block (`internal/app/serveapp/turn_timings.go`), the cost counters (`internal/cost`), and this census.
- **Not done:**
  - The prior-art reading `docs/tasks/task-forward-architecture-2026-10.md` §10a lists as not read (TVM and Relax, FlashInfer's and FlashAttention's kernel bodies, the mixed-batch internals, vLLM's model runner). It needs the sources, not a run.
  - The "at what cost" column of the census: each decline's price is the CPU-against-GPU decode ratio of that family, which this record does not re-measure (the 10-13 baseline quoted about 15× from other records). The cost counters record it per turn for a run that has the GPU path available to compare.
  - A WebGPU census by operation, which waits for OP4 on WebGPU.
