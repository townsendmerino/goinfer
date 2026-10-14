# Forward-architecture baseline: what tokenizing costs, and why each family decodes on the CPU (2026-10-13)

**Status: exploratory.** Two readings taken for `docs/tasks/task-forward-architecture-2026-10.md` (§1 and §3.2), at main
`9593bd70`, on the MacBook (Apple M1 Pro, 16 GB, go1.27.0 darwin/arm64). Neither was registered, neither is a gate, and
neither decides anything alone. The programs are below, so either can be re-run. §3 was added the same day by OP0: a
correctness check on Metal, before and after OP0's activation fix.

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
  neither is expected to cost more than the encode (INFERRED). Not checked either: a tool-heavy agent request, whose tool
  schemas go through `chat/tools.go:Template.RenderTools`. Per-turn timing on the wire (the task doc's RF0) would show it.

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
- `representativeConfig("nemotron_h")` is dense (`MoE == nil`), so the matrix shows it resident on WebGPU. A Nemotron-H MoE
  checkpoint (Nemotron 3 Nano) fails `decoder/residency.go:Architecture.decodeRunnerEligible` on CUDA and Metal, and WebGPU
  admits it through `decoder/residency.go:Model.decodeRunnerDecline` (`docs/hardware-matrix.md`'s note).

**The runner-shape lines, broken down by a second test** (its output is the second block):
- **Llama 4 and LFM2** run their own forwards, which no GPU runner bridges.
- **Granite-4.0-H** is held by its bring-up gate (`GOINFER_SSM_RESIDENT` empty).
- **Laguna and Spark-X2.5 are not own forwards.** They fail `decoder/residency.go:Architecture.ropeResidentCompatible`: their
  local and global RoPE tables have different lengths. In the representative configs these are 8 and 4 for Laguna, and 16 and
  4 for Spark-X2.5.

**What the runner-shape line hides** (third test, added 2026-10-13 after a review, run at `2e760f25`, which changed only docs
since `9593bd70`). The first test prints only the runner-shape reason when that check refuses, so it hides what
`decoder/features.go:residentGateReason` would also say. Printed separately (third block):
- **LFM2** also needs `short-conv`, on all three backends.
- **Granite-4.0-H** also needs `ssm` on CUDA and Metal. On WebGPU the bring-up gate is the only reason.
- **Laguna and Spark-X2.5** also need `attn-output-gate` (Laguna's softplus gate, Spark-X2.5's sigmoid), on all three.
- **Llama 4** has no feature reason on any backend. Only the runner shape stops it.
- **A hazard, not a decline: Spark-X2.5's activation.**
  - Spark-X2.5 is a gated MLP with exact-erf GELU (`ActGelu`, `gegluExact` in `decoder/mlp.go`).
  - `residentFeatures` maps any gated non-SiLU activation to `gated-gelu`, which every backend implements as tanh-GELU.
  - `decoder/residency.go:Model.GatedActResident` passes the activation's ordinal (3 for `ActGelu`) to a CUDA kernel that knows
    0 (tanh-GELU) and 1 (SiLU). `decoder/arch.go`'s comment on `ActGelu`, "Only non-gated archs reach it today", predates
    Spark-X2.5.
  - Nothing runs wrong today: the runner shape and `attn-output-gate` decline Spark-X2.5 first. Once both are lifted, it would
    be admitted with the wrong activation, so the feature set needs an exact-GELU entry before then (read from code, not run).

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

```
granitemoehybrid cuda   runnerOK=false gate="cuda does not implement [ssm], which this model needs"
granitemoehybrid metal  runnerOK=false gate="metal does not implement [ssm], which this model needs"
granitemoehybrid webgpu runnerOK=false gate=""
granitemoehybrid Act=SiLU
laguna           cuda   runnerOK=false gate="cuda does not implement [attn-output-gate], which this model needs"
laguna           metal  runnerOK=false gate="metal does not implement [attn-output-gate], which this model needs"
laguna           webgpu runnerOK=false gate="webgpu does not implement [attn-output-gate], which this model needs"
laguna           Act=SiLU
lfm2             cuda   runnerOK=false gate="cuda does not implement [short-conv], which this model needs"
lfm2             metal  runnerOK=false gate="metal does not implement [short-conv], which this model needs"
lfm2             webgpu runnerOK=false gate="webgpu does not implement [short-conv], which this model needs"
lfm2             Act=SiLU
llama4_text      cuda   runnerOK=false gate=""
llama4_text      metal  runnerOK=false gate=""
llama4_text      webgpu runnerOK=false gate=""
llama4_text      Act=SiLU
spark2_5         cuda   runnerOK=false gate="cuda does not implement [attn-output-gate], which this model needs"
spark2_5         metal  runnerOK=false gate="metal does not implement [attn-output-gate], which this model needs"
spark2_5         webgpu runnerOK=false gate="webgpu does not implement [attn-output-gate], which this model needs"
spark2_5         Act=GELU
nemotron_h       cuda   runnerOK=true gate="cuda does not implement [non-gated-mlp ssm], which this model needs"
nemotron_h       metal  runnerOK=true gate="metal does not implement [ssm], which this model needs"
nemotron_h       webgpu runnerOK=true gate=""
nemotron_h       Act=ReLU²
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

The third test (`decoder/zz_gem_tmp_test.go`, in a second detached worktree, deleted after the run; same command with
`-run '^TestZZGem$'`):

```go
package decoder

import (
	"fmt"
	"testing"
)

func TestZZGem(t *testing.T) {
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	for _, mt := range []string{"granitemoehybrid", "laguna", "lfm2", "llama4_text", "spark2_5", "nemotron_h"} {
		a, _, err := resolveArchitecture(representativeConfig(mt))
		if err != nil {
			t.Fatal(err)
		}
		for _, be := range []string{"cuda", "metal", "webgpu"} {
			fmt.Printf("%-16s %-6s runnerOK=%v gate=%q\n", mt, be, a.decodeRunnerEligible(), residentGateReason(a, be))
		}
		fmt.Printf("%-16s Act=%v\n", mt, a.Act)
	}
}
```

## 3. OP0 follow-up: a non-gated exact-GELU model on Metal, before and after the fix

**Status: a correctness check, not a timing.** Run once on the MacBook (Apple M1 Pro, 16 GB) on 2026-10-13, with no timed
job holding the lock. It was taken while doing OP0, after reading the code showed the gated hazard of §2 had a non-gated
twin that nothing masked.

**Result:**
- **Before the fix** (main `bf62dd71`), Metal built a resident for a GPT-2 checkpoint that declares the exact `"gelu"`, and
  ran the tanh approximation: its logits were bit-identical to the `"gelu_new"` config's. The CPU's logits for the two
  configs differed.
- **After the fix** (branch `op0-2026-10`), the same load declines by name and decodes on the CPU.

**What ran:** a temporary test in package `metal`, not committed, built with `-tags goinfer_testhooks`.
- The two checkpoints are the real GPT-2 (the gitignored fixture Metal's GPT-2 parity test loads) and a copy whose `config.json` says
  `"activation_function": "gelu"`, with the weights symlinked. Only the declared activation differs.
- Each loads at `--quant int8int8` on Metal and on the CPU, and runs the same 8 tokens.
- The comparison is exact equality of the logits. Two configs that run the same kernel give identical bits, so no
  tolerance is needed.
- The command, from each tree: `go test -tags goinfer_testhooks ./metal -run '^TestZZGeluErfScratch$' -count=1 -v`.

Before, at `bf62dd71`:

```
metal gelu_new: metal-resident (int8int8)
metal gelu:     metal-resident (int8int8)
metal gelu vs metal gelu_new: maxAbs=0 bit-identical=true
cpu   gelu vs cpu   gelu_new: maxAbs=5.26986 bit-identical=false
```

After, on `op0-2026-10` (the `-1` is the program's "no resident to compare"):

```
metal gelu_new: metal-resident (int8int8)
metal gelu:     NOT RESIDENT: cpu (int8int8) — requested metal → running on cpu: metal does not implement [non-gated-gelu-erf], which this model needs, and metal has no staged decode path | decline: metal does not implement [non-gated-gelu-erf], which this model needs
metal gelu vs metal gelu_new: maxAbs=-1 bit-identical=false
cpu   gelu vs cpu   gelu_new: maxAbs=5.26986 bit-identical=false
```

**What it does not show:**
- **How wrong the output was.** The CPU's two configs differ by up to 5.27 in a logit over these 8 tokens, at int8int8 on
  weights trained for `gelu_new`. That is the size of the difference between the two activations in this configuration, not a
  measured error against a reference for a checkpoint trained with the exact function. None was at hand.
- **The gated case.** Spark-X2.5 never reaches a GPU, so its hazard is read from the code and pinned by the decoder's tests
  (`TestGatedAct_onlyTanhAndSiLUAdmitted`, `TestGatedGELUErf_sparkNamesIt`), not run on a device.

The test file (`metal/zz_gelu_erf_scratch_test.go`, deleted after each run):

```go
//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Scratch (not committed): does Metal build a resident for a GPT-2 checkpoint declaring the exact "gelu", and which
// activation does it run? Resident logits bit-identical across the "gelu" and "gelu_new" configs while the CPU's differ
// means the resident ran the same (tanh) activation for both.
func TestZZGeluErfScratch(t *testing.T) {
	const tanhDir = "/Users/francistownsend-merino/tmcode/goinfer/testdata/gpt2"
	const erfDir = "<scratch>/gpt2-erf"
	run := func(dir, backend string) ([][]float32, string) {
		m, err := decoder.Load(dir, decoder.Options{Backend: backend, Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load %s %s: %v", backend, dir, err)
		}
		defer m.Close()
		_, _, _, _, _, _, vocab := m.Dims()
		var out [][]float32
		if backend == "cpu" {
			cache := m.NewCache(9)
			for i := range 8 {
				l, err := m.ForwardForTest((i*131+7)%vocab, cache)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), l...))
			}
			return out, m.DecodePath()
		}
		rf := m.ResidentForwardForTest()
		if rf == nil {
			return nil, "NOT RESIDENT: " + m.DecodePath() + " | decline: " + m.ResidentDecline()
		}
		rf.Reset()
		for i := range 8 {
			l, err := rf.Forward(m.EmbedResidentForTest((i*131+7)%vocab), i)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out, m.DecodePath()
	}
	maxDiff := func(a, b [][]float32) (float64, bool) {
		if a == nil || b == nil {
			return -1, false
		}
		md, same := 0.0, true
		for i := range a {
			for j := range a[i] {
				d := float64(a[i][j] - b[i][j])
				if d < 0 {
					d = -d
				}
				if d > md {
					md = d
				}
				if a[i][j] != b[i][j] {
					same = false
				}
			}
		}
		return md, same
	}
	mt, pt := run(tanhDir, "metal")
	me, pe := run(erfDir, "metal")
	ct, _ := run(tanhDir, "cpu")
	ce, _ := run(erfDir, "cpu")
	fmt.Printf("metal gelu_new: %s\nmetal gelu:     %s\n", pt, pe)
	d, same := maxDiff(mt, me)
	fmt.Printf("metal gelu vs metal gelu_new: maxAbs=%.6g bit-identical=%v\n", d, same)
	d, same = maxDiff(ct, ce)
	fmt.Printf("cpu   gelu vs cpu   gelu_new: maxAbs=%.6g bit-identical=%v\n", d, same)
}
```
