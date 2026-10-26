//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// ksh6cModel is one of H6C-G2/G3's loads: a checkpoint under ~/models, a pinned context (0: unpinned) and the expert cache.
type ksh6cModel struct {
	rel      string
	ctx      int
	moeCache bool
}

// H6C-G2's models (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build: gates"), one test per heavy-tier group, since a process
// holds at most one large checkpoint (cmd/gate's heavyFamilies). KSH6C_MODELS (a comma-separated list of file names) narrows each.

func TestKSH6C_marginPriced(t *testing.T) {
	ksh6cMarginPriced(t, []ksh6cModel{
		{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 0, false},
		{"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 0, false},
		{"gemma-3-4b-it", 0, false},
		{"qwen25vl-3b-instruct", 0, false},
		{"gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf", 0, false},
	})
}

func TestKSH6C_marginPriced7B8B(t *testing.T) {
	ksh6cMarginPriced(t, []ksh6cModel{
		{"qwen2.5-7b-instruct-q4_k_m.gguf", 0, false},
		{"qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf", 16384, false},
	})
}

func TestKSH6C_marginPriced26B(t *testing.T) {
	ksh6cMarginPriced(t, []ksh6cModel{{"gemma4-26b-q4_k_m.gguf", 0, true}})
}

// ksh6cPageCharged is postPoolBytes' rule applied to observed buffers: those under the quantum share whole pages, each larger one
// is rounded up to the quantum.
func ksh6cPageCharged(bufs []int) int64 {
	var small, large int64
	for _, b := range bufs {
		if n := int64(b); n < allocQuantumBytes {
			small += n
		} else {
			large += chargedBytes(n)
		}
	}
	return (small+allocQuantumBytes-1)/allocQuantumBytes*allocQuantumBytes + large
}

// ksh6cMarginPriced is H6C-G2 and H6C-G3, with H6C-G5's after-figures. Each model loads through decoder.Load as serve loads it,
// after a warm-up load that runs the startup self-test (whose pool the plan then sees), with every device allocation the build makes
// recorded in order.
//   - G2: the buffers between the pools and the self-sizing grants, rounded by postPoolBytes' rule, equal postPoolBytes on the built
//     resident; the pools' buffers, each charged, equal kvChargedBytes at the built context and slots; and the margin recomputed on
//     the built resident at the context the plan priced it at equals the one the plan kept (the planned geometry was the built one).
//   - G3, on an unpinned load without the expert cache: the build kept the planned context, and the planned context is at most 64
//     positions below the largest the build's exact figure allows at its fit-check reading, beyond the rounding bound and Plan's own
//     dense-weight residual (the device's drop from before the load to the fit check, less Plan's dense figure and the packing slack,
//     which is Plan's and not the margin's, and is reported).
//
// A pinned load the computed margin refuses is H6C-G5's reading (a configuration that loads today and is refused after), not a
// failure: it is printed and the model is skipped.
func ksh6cMarginPriced(t *testing.T, models []ksh6cModel) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("KSH6C_MARGIN") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and KSH6C_MARGIN=1")
	}
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	// Free VRAM is read through a short-lived device (decoder.FreeBytesFor): a handle held across the loads would keep the primary
	// context, and the self-test's local-memory pool, alive where serve keeps neither.
	free := func() int64 {
		f, ok := decoder.FreeBytesFor("cuda")
		if !ok {
			t.Fatal("no free-VRAM reading")
		}
		return f
	}
	want := map[string]bool{}
	if v := os.Getenv("KSH6C_MODELS"); v != "" {
		for _, s := range strings.Split(v, ",") {
			want[strings.TrimSpace(s)] = true
		}
	}
	kvF16Auto.Store(true) // serve's --kv auto: f16 KV where the family is cleared
	defer kvF16Auto.Store(false)
	if w, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"}); err == nil {
		_ = w.Close() // the startup self-test runs here, once, before any figure below is read
	}
	ran := 0
	for _, c := range models {
		if len(want) > 0 && !want[filepath.Base(c.rel)] {
			continue
		}
		t.Run(filepath.Base(c.rel), func(t *testing.T) {
			path := filepath.Join(home, "models", c.rel)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no model at %s", path)
			}
			var (
				mu                sync.Mutex
				phase             int // 0 before the pools, 1 the pools, 2 between the pools and the grants, 3 after
				pools, post       []int
				fitFree, before   int64
				poolsSeen, gotFit bool
				lastPre           int
			)
			allocSeenForTest = func(b int) {
				mu.Lock()
				defer mu.Unlock()
				switch phase {
				case 0:
					lastPre = b
				case 1:
					pools = append(pools, b)
				case 2:
					post = append(post, b)
				}
			}
			kvAllocSeenForTest = func(_ *cudaResident, b int) {
				mu.Lock()
				defer mu.Unlock()
				if phase == 0 {
					phase, poolsSeen = 1, true
					if lastPre == b { // f16 (akv): this buffer's allocSeen fired first, in phase 0; f32's fires after, in phase 1
						pools = append(pools, b)
					}
				}
			}
			kvPoolsDoneForTest = func() { mu.Lock(); phase = 2; mu.Unlock() }
			postPoolDoneForTest = func() { mu.Lock(); phase = 3; mu.Unlock() }
			saved := cudaFreeVRAM
			cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
				f, e := saved(r)
				if !gotFit {
					fitFree, gotFit = int64(f), true
				}
				return f, e
			}
			defer func() {
				allocSeenForTest, kvAllocSeenForTest, kvPoolsDoneForTest, postPoolDoneForTest, cudaFreeVRAM = nil, nil, nil, nil, saved
			}()
			before = free()
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: c.ctx, MoECacheExperts: c.moeCache})
			allocSeenForTest, kvAllocSeenForTest, kvPoolsDoneForTest, postPoolDoneForTest, cudaFreeVRAM = nil, nil, nil, nil, saved
			if err != nil && c.ctx > 0 && strings.Contains(err.Error(), "GB of KV") {
				fmt.Fprintf(os.Stderr, "[H6C-G5] %s at -ctx %d: REFUSED after the change: %v\n", c.rel, c.ctx, err)
				return
			}
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			ran++
			if !poolsSeen {
				t.Fatal("no K/V pool allocation seen")
			}
			mib := func(b int64) float64 { return float64(b) / (1 << 20) }
			gotPost, wantPost := ksh6cPageCharged(post), r.postPoolBytes()
			var gotKV int64
			for _, b := range pools {
				gotKV += chargedBytes(int64(b))
			}
			wantKV := kvChargedBytes(r.ctxCap, r.kvSlotsN, r.layers)
			marginNow := r.ctxMarginFor(r.ctxMarginCtx)
			fmt.Fprintf(os.Stderr, "[H6C-G2] %s: post-pool %d buffers, %.2f MiB charged against postPoolBytes %.2f MiB; pools %d buffers, %.2f MiB charged against kvChargedBytes %.2f MiB; margin %.1f MiB planned, %.1f MiB on the built resident (at %d positions)\n",
				c.rel, len(post), mib(gotPost), mib(wantPost), len(pools), mib(gotKV), mib(wantKV), mib(r.ctxMarginBytes), mib(marginNow), r.ctxMarginCtx)
			if gotPost != wantPost {
				t.Errorf("G2: the build allocated %d B after the pools (charged), postPoolBytes prices %d B", gotPost, wantPost)
			}
			if gotKV != wantKV {
				t.Errorf("G2: the pools charged %d B, kvChargedBytes prices %d B", gotKV, wantKV)
			}
			if marginNow != r.ctxMarginBytes {
				t.Errorf("G2: the margin on the built resident is %d B, the plan kept %d B: the planned geometry was not the built one", marginNow, r.ctxMarginBytes)
			}
			// H6C-G5's after-figures.
			fmt.Fprintf(os.Stderr, "[H6C-G5] %s: context %d (planned %d), %d KV slots, margin %.1f MiB\n", c.rel, r.ctxCap, r.ctxPlanned, r.kvSlotsN, mib(r.ctxMarginBytes))
			if c.ctx != 0 || c.moeCache {
				return
			}
			if r.ctxCap != r.ctxPlanned {
				t.Errorf("G3: the build trimmed the planned context %d to %d", r.ctxPlanned, r.ctxCap)
			}
			ceiling := fitDefaultCtx
			if mc := m.Config().MaxPositions; mc > 0 && mc < ceiling {
				ceiling = mc
			}
			exact, ok := kvChargedCapForBytes(fitFree-r.extraBytes-r.ctxMarginBytes, r.kvSlotsN, r.ctxPlanned, ceiling, r.layers)
			if !ok {
				exact = r.ctxPlanned
			}
			perPos := float64(kvBytesForCap(kvBlock, r.layers)) / kvBlock
			owners, mla := kvLayerCounts(r.layers)
			round := float64(kvRoundingBound(owners, mla)) / perPos
			residual := float64((before-fitFree)-(m.ResidentDenseWeightBytesFor("cuda")+r.allocSlackBytes)) / perPos
			allowed := 64 + round + max(0, -residual)
			fmt.Fprintf(os.Stderr, "[H6C-G3] %s: planned %d, the build's exact figure allows %d (ceiling %d); the rounding bound %.0f positions, Plan's dense residual %.0f positions; %d below, %.0f allowed\n",
				c.rel, r.ctxPlanned, exact, ceiling, round, residual, exact-r.ctxPlanned, allowed)
			if float64(exact-r.ctxPlanned) > allowed {
				t.Errorf("G3: the planned context %d is %d positions below the build's exact allowance %d (allowed %.0f)", r.ctxPlanned, exact-r.ctxPlanned, exact, allowed)
			}
		})
	}
	if ran == 0 {
		t.Skip("no model built a CUDA resident here")
	}
}
