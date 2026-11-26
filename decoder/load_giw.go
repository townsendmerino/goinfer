package decoder

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/internal/runtime/giw"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// loadGIW is Load for a prequant bundle (.giw): the weights are already quantized and serialized, so
// they alias straight out of the file — no GGUF/safetensors load, no requant
// (opts.Quant/LoRA do not apply). The file is mmap'd read-only so the aliased
// int8/int4 weights are pageable (faulted from the page cache, evictable) rather
// than copied to the heap — the substrate the streaming / expert-paging policies
// build on. giw.Read splits the weight blob from the metadata-GGUF tokenizer; the
// mapping is held on the Model and released by Close.
func loadGIW(dir string, opts Options, be Backend, beErr error, ctxFromCaller bool) (*Model, error) {
	if opts.LoRA != "" {
		closeBackend(be)
		return nil, fmt.Errorf("decoder: LoRA merge needs a safetensors base; %s is a prequantized .giw with no base to merge into", dir)
	}
	data, rerr := mapGIW(dir) // MAP_SHARED on darwin, aikit's MAP_PRIVATE elsewhere (giwmap_*.go)
	if rerr != nil {
		closeBackend(be)
		return nil, fmt.Errorf("decoder: mmap .giw: %w", rerr)
	}
	// Keep the weights out of any fork()ed child (forkinherit_darwin.go): once a GPU backend has wired a page of this private
	// mapping, a fork would otherwise copy all of it eagerly. Non-fatal: failing it only restores the old behaviour.
	if perr := protectGIWMapping(data); perr != nil {
		logs.Warnf("decoder: minherit(VM_INHERIT_NONE) on the .giw mapping failed (%v) — a fork of this process may copy the whole mapping", perr)
	}
	weightsBlob, _, gerr := giw.Read(data)
	if gerr != nil {
		_ = unmapGIW(data)
		closeBackend(be)
		return nil, fmt.Errorf("decoder: parse .giw bundle: %w", gerr)
	}
	// The trailing CRC reads every byte of the mapping, which dominates the load of a large streamed .giw. It is a property of the
	// file, so check it once per (size, mtime) and skip it on later loads (giwverify.go). The stat is taken before the load so a
	// file replaced mid-load cannot inherit the marker.
	gfi, statErr := os.Stat(dir)
	crcDone := statErr == nil && giwVerified(dir, gfi)
	afterWalk := func() {}
	w, lerr := loadSerializedWeightsAbortable(weightsBlob, crcDone, func(a *Architecture) {
		afterWalk = prepareColdLoad(dir, data, a, opts, be)
	}, opts.LoadAbort)
	afterWalk()
	if lerr != nil {
		_ = unmapGIW(data)
		closeBackend(be)
		if errors.Is(lerr, errCopiedF32Bundle) {
			return nil, fmt.Errorf("%s: %w", filepath.Base(dir), lerr)
		}
		return nil, lerr
	}
	if statErr == nil && !crcDone {
		markGIWVerified(dir, gfi)
	}
	// A .giw's weights are file-backed (no fitCheckFor call here, by design; see fitguard.go's srcFileBytes), but its KV cache and
	// prefill scratch are real anonymous allocations. guardGIWFit refuses or auto-pins like the .gguf path's guardFit, against a
	// flat margin over live available memory rather than fitMemFraction's 70%-of-available (sized for a load that commits its
	// weights too). Not everything a .giw load allocates is file-backed: the paged MoE scale cache and the rest of the heap are
	// anonymous, and nothing here prices them.
	if pinnedCtx, gerr := guardGIWFit(&w.Cfg, opts); gerr != nil {
		_ = unmapGIW(data)
		closeBackend(be)
		return nil, gerr
	} else if pinnedCtx > 0 {
		opts.ResidentContext = pinnedCtx
	}
	// A .giw bakes its int4 representation in at write time (giwWriter.target), so unlike a GGUF/safetensors load, where
	// wantsCanonicalInt4 decides from this opts.Backend before a byte is quantized, the reader must check the file's promise
	// against what this Load needs. giwReader.weightMat already refuses a kind-5 tensor
	// this core cannot run (wrong arch/shape); this catches the other mismatch, a kind-5 file under a backend that needs canonical
	// bytes (e.g. Metal), which withResidency's decline would not fail loudly for (it logs and falls back to CPU/staged,
	// non-fatal by design).
	//
	// The zero Backend is the CPU here, as Options.Backend documents. wantsCanonicalInt4 keeps "" conservative for the GGUF
	// paths, where canonical bytes can still be built at load; a .giw's bytes are fixed, and the plain CPU backend runs
	// row4-only weights. A resident build on such a model still refuses by name.
	giwBackend := opts.Backend
	if giwBackend == "" {
		giwBackend = "cpu"
	}
	if wantsCanonicalInt4(giwBackend, be) {
		if n := repackedOnlyInt4Count(w); n > 0 {
			_ = unmapGIW(data)
			closeBackend(be)
			return nil, fmt.Errorf("decoder: %s: %d int4 tensor(s) are stored row4-only (kind 5, a cpu-arm64 prequant target) but Backend %q needs canonical bytes — rebuild with `go run ./cmd/prequant -target <matching this backend>` (or delete the stream-weights cache so it rebuilds automatically)", dir, n, opts.Backend)
		}
	}
	if beErr != nil {
		logs.Warnf("%v", beErr)
	}
	if opts.ActQuantGroup == 0 && w.quantLabel() == "q4k" {
		opts.ActQuantGroup = 32 // a q4k bundle runs per-32 activations, as a direct q4k load does (modelFromOptions)
	}
	m := modelFromOptions(w, be, opts)
	m.resCtxPinned = ctxFromCaller
	m.mmap, m.srcPath, m.loadPath = data, dir, dir
	m.bindKnobs(opts.Knobs.values())
	m.withBackendNames(opts.Backend, beErr)
	if opts.StreamWeights {
		// Resolve an "auto" (0) weight-cache request from this platform's own live probe before either pager sees it, so darwin gets
		// a real figure instead of aikit's Linux-only /proc probe plus a fixed darwin fallback.
		opts.WeightCacheBytes = resolveWeightCacheBudget(opts.WeightCacheBytes)
		// MoE → expert demand-paging (#2); dense → per-layer streaming (#4).
		if w.arch.MoE != nil {
			if m.pager = newExpertPager(w, data, opts.WeightCacheBytes, dir, resolveMoEPagerPool(opts.MoEPager, loadKnob(opts, knobMoEPreadCPU))); m.pager != nil {
				logs.Infof("%s", "decoder: "+pagerSummary(m.pager))
				// Predict the working-set rate and require an explicit acknowledgement below the registered floor, rather than let a long
				// run discover it.
				if predicted, ok := moeWorkingSetPrediction(m); ok {
					logs.Infof("decoder: predicted paged-MoE decode rate ~%.2f tok/s "+
						"(a prior, not a measurement — see moeHitRatePrior's own doc comment)", predicted)
					if rerr := moeWorkingSetRefusal(filepath.Base(dir), predicted, opts.AcceptSlowMoE); rerr != nil {
						m.Close()
						return nil, rerr
					}
				}
			} else {
				logs.Warnf("decoder: --stream-weights ignored (no mmap-backed MoE experts to page)")
			}
		} else if m.layerPager = newLayerPager(w, data, opts.WeightCacheBytes); m.layerPager != nil {
			logs.Infof("%s", "decoder: "+layerPagerSummary(m.layerPager))
		} else {
			logs.Warnf("decoder: --stream-weights ignored (model fits the budget, or no mmap-backed layer weights)")
		}
	}
	m, rerr = m.withResidencyOrAbort()
	if rerr != nil {
		return nil, rerr
	}
	// A MoE load that decodes from its mapping on the CPU (no -stream-weights, no resident runner) pages its experts through
	// the page cache, so it takes the paged load's working-set refusal at the
	// budget a pager would take here.
	if !opts.StreamWeights && w.arch.MoE != nil && !m.ResidentActive() {
		if predicted, ok := mappedMoEWorkingSetPrediction(m, resolveWeightCacheBudget(opts.WeightCacheBytes)); ok {
			logs.Infof("decoder: predicted mapped-MoE decode rate ~%.2f tok/s "+
				"(a prior, not a measurement — see moeHitRatePrior's own doc comment)", predicted)
			if serr := moeWorkingSetRefusal(filepath.Base(dir), predicted, opts.AcceptSlowMoE); serr != nil {
				m.Close()
				return nil, serr
			}
		}
	}
	return m, nil
}
