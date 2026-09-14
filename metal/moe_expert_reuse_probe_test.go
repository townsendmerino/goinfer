//go:build darwin

package metal

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMoEExpertReuseProbe investigates M-05 (audit-metal-2026-09-12.md) cheaply, BEFORE building
// the expert-major prefill restructuring it proposes: on a REAL paged MoE resident, run genuine
// text through the EXACT path paged MoE prefill already uses today — prefillOK is hard-false for
// a paged generic MoE (metal/model.go), so every prompt token already goes through the sequential
// per-token decode loop, one router readback + stage per token per layer — and read off the
// expert pool's own telemetry (stages / distinctExperts / nE) to see whether an expert-major
// regroup (stage each routed expert ONCE per prefill instead of once per token that routes to it)
// would converge close to nE (little win left — routing already touches most experts anyway) or
// stay far below today's stage count (real cache churn from LRU thrashing, so M-05 is worth
// building).
//
// Manual/one-off by design (a real multi-GB checkpoint, several minutes of GPU time) — gated
// behind GOINFER_MOE_REUSE_PROBE_CKPT rather than GOINFER_HEAVY_TESTS' usual asset registry, same
// convention as TestMoEPrefillMeasure_batchedVsSequential (moe_prefill_measure_test.go). Unlike
// that test, this one REQUESTS Metal's own GPU expert-cache paging (Options.MoECacheExperts) with
// slots left at 0 (auto-sized, M-13) rather than requiring the whole expert set resident — the
// qwen15-moe-a27b shape this repo already has on disk needs ~16.7 GB resident non-paged, which
// does not fit a 16 GB Mac at all (confirmed 2026-09-10, ~/models/moe_prefill_measure.log), but
// paged only holds N << nE experts per layer.
//
//	GOINFER_MOE_REUSE_PROBE_CKPT=~/models/qwen15-moe-a27b GOINFER_MOE_REUSE_PROBE_M=512 \
//	  go test -tags metal ./metal/ -run TestMoEExpertReuseProbe -v -timeout 30m
func TestMoEExpertReuseProbe(t *testing.T) {
	ckpt := os.Getenv("GOINFER_MOE_REUSE_PROBE_CKPT")
	if ckpt == "" {
		t.Skip("GOINFER_MOE_REUSE_PROBE_CKPT not set — this is a manual measurement, not a CI gate")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	M := 512
	if v := os.Getenv("GOINFER_MOE_REUSE_PROBE_M"); v != "" {
		if n, err := fmt.Sscanf(v, "%d", &M); err != nil || n != 1 {
			t.Fatalf("GOINFER_MOE_REUSE_PROBE_M=%q: not an int", v)
		}
	}

	// Real text, not random token ids: routing is a function of the hidden state, and a genuine
	// coherent sequence's hidden-state trajectory is not the same statistical object as M
	// independent random-token embeddings. Use this repo's own audit doc — real, on hand, and
	// exactly the kind of long-technical-document prompt this repo's own agentic use case sends.
	tk, err := tokenizer.Load(ckpt)
	if err != nil {
		t.Fatalf("tokenizer.Load: %v", err)
	}
	corpus, err := os.ReadFile("../docs/audit-metal-2026-09-12.md")
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	ids, err := tk.Encode(string(corpus), true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) < M {
		// Repeat the corpus rather than fail — the exact CONTENT matters far less here than
		// having M positions of real-token statistics to route on.
		base := append([]int(nil), ids...)
		for len(ids) < M {
			ids = append(ids, base...)
		}
	}
	ids = ids[:M]

	fmt.Fprintf(os.Stderr, "[moe-reuse-probe] loading %s (int4, metal, MoECacheExperts, auto slots)...\n", ckpt)
	tLoad := time.Now()
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4", MoECacheExperts: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	fmt.Fprintf(os.Stderr, "[moe-reuse-probe] loaded in %s\n", time.Since(tLoad).Round(time.Millisecond))

	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	if r.moe == nil {
		t.Fatal("no generic MoE on this resident — checkpoint is not what this probe needs")
	}
	if !r.moe.paged {
		t.Fatal("resident came up non-paged (fits resident?) — this probe needs the paged expert " +
			"pool this finding is actually about; re-run without MoECacheExperts for the non-paged case")
	}
	nLayers, nPagedLayers := 0, 0
	for l := range r.layers {
		nLayers++
		if r.layers[l].moe != nil && r.layers[l].moe.pool != nil {
			nPagedLayers++
		}
	}
	fmt.Fprintf(os.Stderr, "[moe-reuse-probe] resident: %d layers (%d paged MoE), nE=%d top-%d, hidden %d, M=%d real tokens\n",
		nLayers, nPagedLayers, r.moe.nE, r.moe.k, r.H, M)

	// Sequential pass: the EXACT path a prompt already takes today (prefillOK is false for paged
	// MoE), one router readback + stage per token per layer. resident.Forward(id, pos) is the
	// wrong entry point here — its forwardLogits/encodeTrunkInto path has no paged branch at all
	// and panics on C-02's guard the moment it reaches a paged MoE layer (confirmed the hard way:
	// this test originally called it and hit exactly that panic). ForwardEmb(emb, pos) is what
	// actually branches to forwardLogitsMoEPaged for a paged generic MoE resident, so this embeds
	// each token itself first, mirroring what Forward does internally for the non-paged case.
	embScratch := make([]float32, r.H)
	tSeq := time.Now()
	for i, id := range ids {
		r.embed.Row(id, embScratch)
		if r.embedScale > 1 {
			for j := range embScratch {
				embScratch[j] *= r.embedScale
			}
		}
		r.ForwardEmb(embScratch, i)
		if i > 0 && i%64 == 0 {
			fmt.Fprintf(os.Stderr, "[moe-reuse-probe] %d/%d tokens, elapsed %s\n",
				i, M, time.Since(tSeq).Round(time.Millisecond))
		}
	}
	seqDur := time.Since(tSeq)
	fmt.Fprintf(os.Stderr, "[moe-reuse-probe] done: %d tokens in %s (%.2f ms/token)\n\n",
		M, seqDur.Round(time.Millisecond), float64(seqDur.Microseconds())/1000/float64(M))

	var totalStages, totalHits, totalCold, totalEvict, totalDistinct, totalNaive int
	for l := range r.layers {
		L := &r.layers[l]
		if L.moe == nil || L.moe.pool == nil {
			continue
		}
		p := L.moe.pool
		distinct := len(p.distinctExperts)
		naive := M * r.moe.k // today's worst case if the LRU pool captured NO reuse at all
		t.Logf("layer %2d: stages=%-5d hits=%-5d coldStarts=%-4d evictions=%-4d distinctExperts=%-3d (of nE=%d)  naive M*k=%d",
			l, p.stages, p.hits, p.coldStarts, p.evictions, distinct, r.moe.nE, naive)
		totalStages += p.stages
		totalHits += p.hits
		totalCold += p.coldStarts
		totalEvict += p.evictions
		totalDistinct += distinct
		totalNaive += naive
	}
	fmt.Fprintf(os.Stderr, "\n[moe-reuse-probe] TOTALS across %d paged layers:\n", nPagedLayers)
	fmt.Fprintf(os.Stderr, "  stages (today's actual re-fetches):        %d\n", totalStages)
	fmt.Fprintf(os.Stderr, "  distinctExperts (expert-major floor):      %d\n", totalDistinct)
	fmt.Fprintf(os.Stderr, "  naive M*k (no reuse at all):                %d\n", totalNaive)
	if totalDistinct > 0 {
		fmt.Fprintf(os.Stderr, "  stages / distinctExperts (churn factor):   %.2fx\n", float64(totalStages)/float64(totalDistinct))
	}
	if totalStages > 0 {
		fmt.Fprintf(os.Stderr, "  distinctExperts / stages (grouping headroom): %.1f%% of today's stages are avoidable re-fetches\n",
			100*(1-float64(totalDistinct)/float64(totalStages)))
	}
	t.Logf("TOTALS: stages=%d distinctExperts=%d naive=%d churnFactor=%.2fx",
		totalStages, totalDistinct, totalNaive, float64(totalStages)/float64(max(totalDistinct, 1)))
}
