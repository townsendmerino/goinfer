# Forward-architecture baseline: what tokenizing costs, and why each family decodes on the CPU (2026-10-13)

**Status: exploratory.** Two readings taken for `docs/tasks/task-forward-architecture-2026-10.md` (§1 and §3.2), at main
`9593bd70`, on the MacBook (Apple M1 Pro, 16 GB, go1.27.0 darwin/arm64). Neither was registered, neither is a gate, and
neither decides anything alone. The programs are below, so either can be re-run.

## 1. The tokenizer on an agent-sized prompt

**Result:** `load 89.410375ms; encode 25327 tokens (90000 chars) best-of-5 24.077542ms (0.95 µs/token)`.

- **What was timed:** goinfer's tokenizer, loaded with `tokenizer.LoadGGUF` from `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
  (the Qwen2.5 BPE). It encoded `docs/ARCHITECTURE.md`, repeated to 90,000 characters, five times in one process, and the
  best time was kept.
- **How it ran:** one run, by day, under the timing lock (`python3 scripts/timing_lock.py run --label fwdarch-tokenizer`). No
  timed job held the lock, and none was running.
- **An earlier run the same afternoon** of the same program shape gave a 100.1 ms load and a 24.68 ms encode (1.0 µs a token).
  Its source was not kept, so it is noted here and not used.
- **Against prefill on the same Mac:** the 1.5B prefills a 3,900-token prompt on Metal in 4,256.6 ms. That is the median time
  to first token through serve, from `docs/measurements/peer-sweep-2026-09-29.md` cell h, quoted in `docs/benchmarks.md` under
  "Apple Silicon Metal prefill". It works out to about 1.09 ms a token.
  - Tokenizing costs about 0.09% of that per token.
  - A longer prompt only lowers the share, because attention grows with depth and tokenizing does not.
- **Not measured: rendering the chat template and parsing the request JSON.** Both are linear passes over the same text, so
  neither is expected to cost more than the encode (INFERRED).

The run used a throwaway module beside the tree: `GOWORK=off go build -o tokbench .`, then
`python3 scripts/timing_lock.py run --label fwdarch-tokenizer -- ./tokbench`.

```
module tokbench

go 1.27.0

require github.com/townsendmerino/goinfer v0.0.0

replace github.com/townsendmerino/goinfer => /Users/francistownsend-merino/tmcode/goinfer-fwdarch
```

```go
// Exploratory: how long goinfer's tokenizer takes on an agent-sized prompt (forward-arch baseline, 2026-10-13).
// Loads the Qwen2.5 tokenizer from a GGUF, repeats docs/ARCHITECTURE.md's text to 90,000 characters, and encodes it
// five times, reporting the best.
package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func main() {
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	tk, err := tokenizer.LoadGGUF(home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err != nil {
		panic(err)
	}
	load := time.Since(t0)
	src, err := os.ReadFile(home + "/tmcode/goinfer-fwdarch/docs/ARCHITECTURE.md")
	if err != nil {
		panic(err)
	}
	var sb strings.Builder
	for sb.Len() < 90000 {
		sb.Write(src)
	}
	text := sb.String()[:90000]
	best := time.Duration(1 << 62)
	n := 0
	for i := 0; i < 5; i++ {
		t := time.Now()
		ids, err := tk.Encode(text, false)
		if err != nil {
			panic(err)
		}
		if d := time.Since(t); d < best {
			best = d
		}
		n = len(ids)
	}
	fmt.Printf("load %v; encode %d tokens (%d chars) best-of-5 %v (%.2f µs/token)\n",
		load, n, len(text), best, float64(best.Nanoseconds())/1e3/float64(n))
}
```

## 2. Why each family decodes on the CPU, per GPU backend

**Result:**
- The registry has 52 keys and 43 distinct `Architecture.Name`s, which are the hardware matrix's 43 rows.
- Families that decode CPU-only: 10 under CUDA, 12 under Metal and 16 under WebGPU.
- Each output line below gives a family, a backend, and the reason the runtime gives for declining.

**What ran:** a temporary test in package `decoder`, not committed.
- It walks the registry and resolves each key's `representativeConfig`, the hardware matrix's own fixtures.
- It dedupes by `Architecture.Name`.
- For each GPU backend where `decoder/features.go:ResidentEligible` is false, it prints the reason:
  - a runner-shape line when `decoder/residency.go:Architecture.decodeRunnerEligible` refuses first;
  - otherwise `decoder/features.go:residentGateReason`.
- `GOINFER_SSM_RESIDENT` and `GOINFER_GEMMA4_RESIDENT` are set empty, as the matrix generator sets them.
- The run command, from a detached worktree at `9593bd70`, was `GOWORK=off go test ./decoder -run '^TestZZDeclineCensus$' -count=1 -v`.

**The configuration it reads** is the matrix's model-free default: `residentGateReason` passes `actSafe = false`. A loaded
model can differ, because `decoder/residency.go:Model.residentAdmission` reads the model's own configuration.
- On CUDA, a Phi-3/4 GGUF loads at `--quant q4k` (`internal/modelload/modelload.go:activationSafeQuant`) and runs resident.
- On Metal it runs resident only at an explicit `--quant q4k`.

**The runner-shape lines, broken down by a second test** (its output is the second block):
- **Llama 4 and LFM2** run their own forwards, which no GPU runner bridges.
- **Granite-4.0-H** is held by its bring-up gate (`GOINFER_SSM_RESIDENT` empty).
- **Laguna and Spark-X2.5 are not own forwards.** They fail `decoder/residency.go:Architecture.ropeResidentCompatible`: their
  local and global RoPE tables have different lengths. In the representative configs these are 8 and 4 for Laguna, and 16 and
  4 for Spark-X2.5.

```
bailing_hybrid   cuda    cuda does not implement [kda], which this model needs   (registry keys: bailing_hybrid)
bailing_hybrid   metal   metal does not implement [kda mla], which this model needs   (registry keys: bailing_hybrid)
bailing_hybrid   webgpu  webgpu does not implement [kda], which this model needs   (registry keys: bailing_hybrid)
cohere           webgpu  webgpu does not implement [layer-norm pairwise-rope parallel-block], which this model needs   (registry keys: cohere)
cohere2          webgpu  webgpu does not implement [layer-norm nope pairwise-rope parallel-block], which this model needs   (registry keys: cohere2)
deepseek_v2      metal   metal does not implement [mla], which this model needs   (registry keys: deepseek_v2)
deepseek_v3      metal   metal does not implement [mla], which this model needs   (registry keys: deepseek_v3)
gemma2           cuda    cuda does not implement [attn-logit-softcap], which this model needs   (registry keys: gemma2)
gemma2           metal   metal does not implement [attn-logit-softcap], which this model needs   (registry keys: gemma2)
gemma2           webgpu  webgpu does not implement [attn-logit-softcap], which this model needs   (registry keys: gemma2)
glm_ocr          webgpu  webgpu does not implement [pairwise-mrope pairwise-rope], which this model needs   (registry keys: glm_ocr, glm_ocr_text)
gpt2             cuda    cuda does not implement [learned-pos non-gated-mlp], which this model needs   (registry keys: gpt2)
gpt2             webgpu  webgpu does not implement [layer-norm learned-pos], which this model needs   (registry keys: gpt2)
granitemoehybrid cuda    runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: granitemoehybrid)
granitemoehybrid metal   runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: granitemoehybrid)
granitemoehybrid webgpu  runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: granitemoehybrid)
kimi_k2          metal   metal does not implement [mla], which this model needs   (registry keys: kimi_k2)
laguna           cuda    runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: laguna)
laguna           metal   runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: laguna)
laguna           webgpu  runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: laguna)
lfm2             cuda    runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: lfm2, lfm2_vl)
lfm2             metal   runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: lfm2, lfm2_vl)
lfm2             webgpu  runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: lfm2, lfm2_vl)
llama4_text      cuda    runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: llama4_text)
llama4_text      metal   runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: llama4_text)
llama4_text      webgpu  runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: llama4_text)
mistral3         webgpu  webgpu does not implement [attn-temp], which this model needs   (registry keys: ministral3, mistral3)
nemotron_h       cuda    cuda does not implement [non-gated-mlp ssm], which this model needs   (registry keys: nemotron_h)
nemotron_h       metal   metal does not implement [ssm], which this model needs   (registry keys: nemotron_h)
olmo3            webgpu  webgpu does not implement [post-only-norm qk-norm-whole], which this model needs   (registry keys: olmo3)
olmo_hybrid      webgpu  webgpu does not implement [nope post-only-norm qk-norm-whole], which this model needs   (registry keys: olmo_hybrid)
phi3             cuda    every resident cuda projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)   (registry keys: phi3)
phi3             metal   every resident metal projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)   (registry keys: phi3)
phi3             webgpu  every resident webgpu projection quantizes activations to int8, and Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)   (registry keys: phi3)
smollm3          webgpu  webgpu does not implement [nope], which this model needs   (registry keys: smollm3)
spark2_5         cuda    runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: spark2_5)
spark2_5         metal   runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: spark2_5)
spark2_5         webgpu  runner shape: own forward not bridged, or an env-gated bring-up   (registry keys: spark2_5)
registry keys 52; distinct Architecture.Name 43; CPU-only: cuda 10, metal 12, webgpu 16
```

```
granitemoehybrid ownForward=true(granitemoehybrid) ropeCompatible=true invFreq local/global=8/8 moe=true
laguna           ownForward=false() ropeCompatible=false invFreq local/global=8/4 moe=true
lfm2             ownForward=true(lfm2) ropeCompatible=true invFreq local/global=8/8 moe=false
llama4_text      ownForward=true(llama4_text) ropeCompatible=true invFreq local/global=4/4 moe=true
spark2_5         ownForward=false() ropeCompatible=false invFreq local/global=16/4 moe=false
```

The test file (`decoder/zz_census_tmp_test.go` in the detached worktree, deleted after the run):

```go
package decoder

// Temporary (not committed): why each registry family runs CPU-only on each GPU backend, in the default configuration.
// Same pieces as the hardware matrix: representativeConfig -> resolveArchitecture -> decodeRunnerEligible, then
// residentGateReason. Deduplicated by Architecture.Name.

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
)

func TestZZDeclineCensus(t *testing.T) {
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
	cpuOnly := map[string]int{}
	for _, name := range order {
		arch := archs[name]
		for _, be := range []string{"cuda", "metal", "webgpu"} {
			if ResidentEligible(arch, be) {
				continue
			}
			cpuOnly[be]++
			why := residentGateReason(arch, be)
			if !arch.decodeRunnerEligible() {
				why = "runner shape: own forward not bridged, or an env-gated bring-up"
			}
			fmt.Fprintf(&b, "%-16s %-7s %s   (registry keys: %s)\n", name, be, why, strings.Join(seen[name], ", "))
		}
	}
	fmt.Fprintf(&b, "registry keys %d; distinct Architecture.Name %d; CPU-only: cuda %d, metal %d, webgpu %d\n",
		len(keys), len(order), cpuOnly["cuda"], cpuOnly["metal"], cpuOnly["webgpu"])
	os.Stdout.WriteString(b.String())
}

func TestZZRunnerShapeDetail(t *testing.T) {
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	for _, mt := range []string{"granitemoehybrid", "laguna", "lfm2", "llama4_text", "spark2_5"} {
		arch, _, err := resolveArchitecture(representativeConfig(mt))
		if err != nil {
			t.Fatal(err)
		}
		f, own := arch.ownForward()
		fmt.Printf("%-16s ownForward=%v(%s) ropeCompatible=%v invFreq local/global=%d/%d moe=%v\n", mt, own, f.Name,
			arch.ropeResidentCompatible(), len(arch.ropeInvFreqLocal), len(arch.ropeInvFreqGlobal), arch.MoE != nil)
	}
}
```
