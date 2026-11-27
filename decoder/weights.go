package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"

	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

// repackedOnlyInt4Count reports how many of w's matmulWeights() tensors are int4-resident with no canonical bytes
// (IsInt4() true, Int4()'s ok false): a kind-5 .giw tensor, or an in-RAM repacked-only build that reached a .giw
// round-trip (no writer produces one today). decoder.Load's .giw branch uses it to refuse such a file under a backend
// that needs canonical bytes.
func repackedOnlyInt4Count(w *darch.Weights) int {
	n := 0
	for _, m := range w.MatmulWeights() {
		if !m.IsInt4() {
			continue
		}
		if _, _, _, ok := m.Int4F16(); !ok {
			n++
		}
	}
	return n
}

// LoadWeights reads config.json and the model.safetensors blob (or sharded set) from an on-disk directory in the HF
// snapshot layout. The blob is mmapped, not heap-copied, so a large checkpoint stays in the OS page cache. bf16 and f16
// weights are widened to f32 on load, which roughly doubles resident RAM against keeping them bf16; per-tile widening
// inside matmul would avoid that and is not implemented.
//
// Use LoadWeightsFromFS for fs.FS-backed paths (MapFS, embed.FS); those stay heap-backed because fs.FS exposes no file
// descriptor.
func LoadWeights(dir string) (*darch.Weights, error) {
	return loadWeights(dir, weightLoadOpts{needCanonical: true})
}

// loadWeights is the quant-aware internal load. When quant is int8/int4, each
// matmul tensor is quantized the moment it is read and its f32 backing is freed
// before the next tensor loads — so the transient footprint is the quantized
// model plus one tensor's f32, not the whole model in f32. That is what lets a
// big quantized checkpoint load in a quarter (int8) or eighth (int4) of the RAM
// the load-everything-then-quantize path needed. The forward output is identical
// to quantizing after load; only the peak memory differs.
func loadWeights(dir string, o weightLoadOpts) (*darch.Weights, error) {
	quant, embedInt4, needCanonical, skipRow4, lora, abort := o.quant, o.embedInt4, o.needCanonical, o.skipRow4, o.lora, o.abort
	// One atomic add per model load, so the fit guard's test can observe that a refused load allocated nothing rather than
	// infer it from an error string (a guard that fires after the allocation still looks correct that way).
	weightAllocs.Add(1)
	if strings.HasSuffix(dir, ".gguf") {
		return loadGGUFWeights(dir, quant, embedInt4, needCanonical, skipRow4, abort) // quantized llama.cpp checkpoint; LoRA guarded in Load
	}
	if quant == quantInt4Mix {
		return nil, fmt.Errorf("decoder: int4mix is GGUF-only (got safetensors %s)", dir)
	}
	if quant == quantQ4K {
		return nil, fmt.Errorf("decoder: q4k keeps a GGUF's Q4_K tensors as they are, so it is GGUF-only (got safetensors %s)", dir)
	}
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		return nil, err
	}
	arch, schema, err := resolveArchitecture(cfg) // selects + validates the family descriptor
	if err != nil {
		return nil, err
	}
	st, err := openCheckpointMmap(dir)
	if err != nil {
		return nil, err
	}
	// buildWeightsFromSafetensors retains st (the WeightMats may alias its mmap) only on success; on any error return st
	// would leak the mapping and fd. A serve process probing candidate dirs, or retrying a load of a checkpoint with one
	// missing tensor, would accumulate address space, the leak Model.Close exists to avoid.
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, o)
	if err == nil {
		// Every family's loader, not only the generic one, answers for the adapter here: a delta the load never merged is
		// refused, which covers the loaders that take no lora at all.
		err = lora.CheckAllMerged()
	}
	if err != nil {
		_ = st.Close()
		return nil, err
	}
	// Release the source mapping now when nothing can alias it, instead of holding it for the model's whole life. The mapping
	// is the bf16 checkpoint, while the quantized weights decode reads are a separate, much smaller allocation; holding the
	// source lets dead pages compete with hot weights for page cache (higher RSS and slower decode).
	if why := mmapAliasRisk(st); why == "" && !islandsMayAliasSource(w) && os.Getenv("GOINFER_P13_OFF") == "" {
		_ = st.Close()
		w.ST = nil
	}
	return w, nil
}

// mmapAliasRisk reports the first tensor whose dtype could leave a slice ALIASING the mapping,
// or "" when none can and the source is therefore safe to close at end of load.
//
// The rule comes from aikit's reader, not from inspection of checkpoints. BF16 and F16 are widened
// into a fresh slice by TensorF32/SubF32 — aikit's own comment: "the result then does not alias the
// file". Every other dtype is served by reinterpretLE, which takes a "zero-copy view" whenever the
// payload is aligned, and alignment is the common case. So a checkpoint of BF16/F16 tensors leaves
// nothing pointing into the mapping, and any other dtype might.
//
// Deliberately conservative in two ways. It keys on what the FILE holds rather than on which
// tensors this load happened to read, and it does not try to reason about quant mode — int8/int4
// drop the f32 after quantizing while quantNone keeps it (see streamExperts), and encoding that
// interaction here would put a second, subtler rule in a second place. A false "risky" costs the
// old behaviour; a false "safe" is a use-after-free.
func mmapAliasRisk(st *embed.SafetensorsFile) string {
	for _, n := range st.Names() {
		t, err := st.Tensor(n)
		if err != nil {
			return n // header entry we cannot classify: assume the worst
		}
		if riskyDType(t.DType) {
			return n
		}
	}
	return ""
}

// islandsMayAliasSource reports whether a bf16 island may be the checkpoint's own bits: stIsland keeps a BF16 tensor as a
// zero-copy view of the mapping, which mmapAliasRisk's dtype rule does not see, so a load holding any bf16 island keeps the
// source mapped. Conservative: an island islandMat made bf16 on the heap keeps it too.
func islandsMayAliasSource(w *darch.Weights) bool {
	for i := range w.Layers {
		if sc := w.Layers[i].ShortConv; sc != nil {
			for _, m := range []*linalg.WeightMat{&sc.InProj, &sc.OutProj} {
				if _, bf16, ok := m.Half16(); ok && bf16 {
					return true
				}
			}
		}
	}
	return false
}

// riskyDType reports whether a stored dtype can be handed back as a slice aliasing the mapping.
// BF16/F16 are always widened into fresh storage; every other dtype may be a zero-copy view.
func riskyDType(dt string) bool {
	switch dt {
	case "BF16", "F16":
		return false
	default:
		return true
	}
}

const shardIndexFile = "model.safetensors.index.json"

// openCheckpointMmap mmaps the checkpoint weights: the multi-shard set named by model.safetensors.index.json when
// present, else the single model.safetensors. Either way the returned file resolves Tensor() uniformly.
func openCheckpointMmap(dir string) (*embed.SafetensorsFile, error) {
	indexPath := filepath.Join(dir, shardIndexFile)
	if _, err := os.Stat(indexPath); err == nil {
		st, err := embed.OpenSafetensorsShardedMmap(indexPath)
		if err != nil {
			return nil, fmt.Errorf("decoder: open sharded safetensors: %w", err)
		}
		return st, nil
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder: open safetensors: %w", err)
	}
	return st, nil
}

// LoadWeightsFromFS reads config.json and model.safetensors from fsys/dir, validates every tensor's shape against Cfg,
// and returns the populated bundle. It is heap-backed (fs.ReadFile); use LoadWeights for the mmap path on a real
// directory.
func LoadWeightsFromFS(fsys fs.FS, dir string) (*darch.Weights, error) {
	return loadWeightsFromFS(fsys, dir, quantNone)
}

// loadWeightsFromFS is the quant-aware internal counterpart of loadWeights for
// fs.FS-backed paths (see loadWeights for the streaming-quant rationale).
func loadWeightsFromFS(fsys fs.FS, dir string, quant quantMode) (*darch.Weights, error) {
	cfg, err := loadConfig(fsys, path.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		return nil, err
	}
	st, err := openCheckpointFromFS(fsys, dir)
	if err != nil {
		return nil, err
	}
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, weightLoadOpts{quant: quant, needCanonical: true})
	if err != nil {
		_ = st.Close() // st is retained only on success; close it on error so the mapping/fd doesn't leak
		return nil, err
	}
	return w, nil
}

// openCheckpointFromFS is the fs.FS counterpart of openCheckpointMmap (heap):
// sharded when an index.json is present, else the single file.
func openCheckpointFromFS(fsys fs.FS, dir string) (*embed.SafetensorsFile, error) {
	indexPath := path.Join(dir, shardIndexFile)
	if _, err := fs.Stat(fsys, indexPath); err == nil {
		st, err := embed.OpenSafetensorsShardedFromFS(fsys, indexPath)
		if err != nil {
			return nil, fmt.Errorf("decoder: open sharded safetensors: %w", err)
		}
		return st, nil
	}
	st, err := embed.OpenSafetensorsFromFS(fsys, path.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder: open safetensors: %w", err)
	}
	return st, nil
}

// buildWeightsFromSafetensors fills a *Weights from an already-opened SafetensorsFile, shape-validating every tensor
// against Cfg through the family's tensor schema. The heap (fs.FS) and mmap paths share it so the tensor-name and shape
// contract is one edit, not two.
func buildWeightsFromSafetensors(cfg *Config, arch *darch.Architecture, s *archparts.TensorSchema, st *embed.SafetensorsFile, o weightLoadOpts) (*darch.Weights, error) {
	var lm wbuild.LoRAMerger
	if o.lora != nil {
		lm = o.lora
	}
	return wbuild.BuildWeightsFromSafetensorsTo(cfg, arch, s, st, o.quant, o.embedInt4, o.needCanonical, o.skipRow4, lm, nil, "", o.abort, wbuildHooks())
}

// weightLoadOpts is how loadWeights and buildWeightsFromSafetensors build a checkpoint's weights: the quantization,
// whether the embedding goes int4, whether the canonical int4 layout is kept beside a repacked one (needCanonical) or
// the row4 fallback skipped (skipRow4), the LoRA adapter to merge, and the load's abort channel.
type weightLoadOpts struct {
	quant                              quantMode
	embedInt4, needCanonical, skipRow4 bool
	lora                               *loraAdapter
	abort                              <-chan struct{}
}

// errLoadAborted aliases wbuild.ErrLoadAborted so errors.Is(err, ErrLoadAborted) matches across packages.
var errLoadAborted = wbuild.ErrLoadAborted

// fusedExpertsTransposedForTest, when set, reads every fused expert tensor in the row layout whatever its shape.
var fusedExpertsTransposedForTest bool

// streamTablesOn routes a streaming transcode's embedding-shaped tables through streamedTable. Tests turn it off to compare
// against the whole-table write.
var streamTablesOn = true

// streamTableBlockBytes bounds pass two's code buffer.
var streamTableBlockBytes = 32 << 20

// IsDirNoStream reports whether err is StreamTranscodeDir declining a family whose loader does not stream.
func IsDirNoStream(err error) bool {
	_, ok := errors.AsType[wbuild.ErrDirNoStream](err)
	return ok
}

// streamLayerBudgetFallback is the window's memory budget on a host with no available-memory probe.
const streamLayerBudgetFallback = 1 << 30

// streamLayerBudget is the memory a streaming transcode may hold in built, unwritten layers: a quarter of the RAM
// available when the transcode starts.
func streamLayerBudget() int64 {
	if a := HostRAMAvailableBytes(); a > 0 {
		return a / 4
	}
	return streamLayerBudgetFallback
}

// wbuildHooks snapshots decoder's diagnostic and test hook globals into a wbuild.Hooks value.
func wbuildHooks() wbuild.Hooks {
	return wbuild.Hooks{
		FakeQuantScheme:        fakeQuantScheme,
		FakeQuantF32Act:        fakeQuantF32Act,
		FakeQuantExpertsOnly:   fakeQuantExpertsOnly,
		FakeQuantPerRow:        fakeQuantPerRow,
		DisableStreamTables:    !streamTablesOn,
		StreamTableBlockBytes:  streamTableBlockBytes,
		StreamDirSwap:          streamDirSwapForTest,
		Q4KWiden56:             q4kWiden56ForTest,
		FusedExpertsTransp:     fusedExpertsTransposedForTest,
		MambaKeepF32:           mambaKeepF32ForTest,
		MLAInt8:                mlaInt8ForTest,
		MLAProjF32:             mlaProjF32ForTest,
		DisableRow4Repack:      !w4a8Row4RepackEnabled,
		Row4RepackCalls:        row4RepackCallsForTest,
		GGUFRowFilter:          ggufRowFilter,
		GIWF32Kind1:            giwF32Kind1ForTest,
		GIWIslandsF32:          giwIslandsF32ForTest,
		SplitHalfRepackEnabled: w4a8SplitHalfRepackEnabled,
		SplitHalfRepacked:      &w4a8SplitHalfRepacked,
		SplitHalfSkipped:       &w4a8SplitHalfSkipped,
		SplitHalfBytes:         &w4a8SplitHalfBytes,
	}
}
