package decoder

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/diskfree"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/sidecar"
)

type (
	TranscodePhases = sidecar.TranscodePhases
	SidecarState    = sidecar.SidecarState
)

const (
	SidecarMissing = sidecar.SidecarMissing
	SidecarLegacy  = sidecar.SidecarLegacy
	SidecarCurrent = sidecar.SidecarCurrent
)

var (
	freeDiskBytes      = diskfree.Free
	sidecarHostRAM     = HostRAMBytes
	renameGIW          = os.Rename
	transcodeSelfCheck func(path string) error
	sidecarInTests     bool
)

func init() { transcodeSelfCheck = selfCheck }

func sidecarOps() sidecar.Ops {
	return sidecar.Ops{
		StreamGGUF:       StreamTranscodeGGUF,
		StreamDir:        StreamTranscodeDir,
		ResidentDirBody:  residentDirBody,
		SelfCheck:        func(path string) error { return transcodeSelfCheck(path) },
		HostRAM:          func() int64 { return sidecarHostRAM() },
		VerifyMarkerPath: GIWVerifiedMarkerPath,
		IsDirNoStream:    IsDirNoStream,
		MarkGIWWritten:   MarkGIWWritten,
		OpenGGUFMmap:     OpenGGUFMmap,
		GGUFFileBytes:    GGUFFileBytes,
		TargetForBackend: GIWTargetForBackend,
		HeaderCheck:      GIWHeaderCheck,
		MapGIW:           mapGIW,
		UnmapGIW:         unmapGIW,
		ScanGIWWeights:   scanGIWWeights,
		Rename:           func(o, n string) error { return renameGIW(o, n) },
		FreeDisk:         func(p string) (int64, bool) { return freeDiskBytes(p) },
		GGUFShards:       GGUFShards,
		ParseQuant:       parseQuant,
	}
}

func TranscodeGIW(ctx context.Context, in, out, quant string, embedInt4 bool, target GIWTarget) error {
	return sidecar.Transcode(sidecarOps(), ctx, in, out, quant, embedInt4, target)
}

func TranscodeGIWTimed(ctx context.Context, in, out, quant string, embedInt4 bool, target GIWTarget) (TranscodePhases, error) {
	return sidecar.TranscodeTimed(sidecarOps(), ctx, in, out, quant, embedInt4, target)
}

func TranscodeGIWLoRA(ctx context.Context, dir, adapter, out, quant string, embedInt4 bool, target GIWTarget) error {
	return sidecar.TranscodeLoRA(sidecarOps(), ctx, dir, adapter, out, quant, embedInt4, target)
}

func TranscodeGIWLoRATimed(ctx context.Context, dir, adapter, out, quant string, embedInt4 bool, target GIWTarget) (TranscodePhases, error) {
	return sidecar.TranscodeLoRATimed(sidecarOps(), ctx, dir, adapter, out, quant, embedInt4, target)
}

func GIWAdapterLoRA(adapterDir, modelPath string) (string, error) {
	return sidecar.AdapterLoRA(adapterDir, modelPath)
}

func EnsureCachedGIW(ctx context.Context, ggufPath, quant, backend string, embedInt4 bool) (string, error) {
	return sidecar.EnsureCached(sidecarOps(), ctx, ggufPath, quant, backend, embedInt4)
}

func SidecarPathIfFresh(ggufPath, quant, backend string, embedInt4 bool) (string, bool) {
	return sidecar.PathIfFresh(sidecarOps(), ggufPath, quant, backend, embedInt4)
}

func SidecarPathIfCurrent(src, quant, backend string, embedInt4 bool) (string, SidecarState) {
	return sidecar.PathIfCurrent(sidecarOps(), src, quant, backend, embedInt4)
}

func SidecarPath(ggufPath, quant, backend string, embedInt4 bool) string {
	return sidecar.Path(sidecarOps(), ggufPath, quant, backend, embedInt4)
}

func DefaultToSidecar(directLoad bool) bool { return sidecar.DefaultEnabled(directLoad) }

func GIWWeightsQuant(path string) (string, error) { return sidecar.WeightsQuant(sidecarOps(), path) }

// residentDirBody is the resident directory transcode: Load the whole model at quant, then serialize it. A LoRA merge
// needs it (the merge works on the f32 projections), and so does a family whose safetensors loader does not stream; it is
// also the byte-identity reference for the streamed bytes.
func residentDirBody(w io.Writer, dir, lora, quant string, embedInt4 bool, target GIWTarget, id string) (int64, error) {
	// ResidentContext 1: a transcode runs no request, so it allocates no KV (selfCheck does the same).
	// Unpinned, the fit guard prices the model's full context and can refuse a model whose weights fit.
	m, err := Load(dir, Options{Quant: quant, EmbedInt4: embedInt4, LoRA: lora, ResidentContext: 1, DirectLoad: true})
	if err != nil {
		return 0, fmt.Errorf("load %s (%s): %w", dir, quant, err)
	}
	defer m.Close()
	return SerializeWeightsToForTarget(w, m.Weights(), id, target)
}

// sidecarRoute is what Load opens for src: src itself, or the sidecar .giw the apps open for it (internal/modelload
// routes by the same rules). A .gguf whose sidecar cannot be built is the load's error, naming Options.DirectLoad as the
// apps' refusal names -direct-load; a safetensors directory falls back to the direct load with a note. A q4k .gguf under
// StreamWeights loads directly: its paging is not built for kind 8.
func sidecarRoute(src string, opts Options) (string, error) {
	if opts.DirectLoad || opts.noSelfTest || opts.planResidency || testing.Testing() && !sidecarInTests {
		return src, nil
	}
	backend := opts.Backend
	if backend == "" {
		backend = "cpu" // Options.Backend's default; GIWTargetForBackend reads "" as no target
	}
	ensure := func() (string, error) {
		return EnsureCachedGIW(context.Background(), src, opts.Quant, backend, opts.EmbedInt4)
	}
	gguf := strings.HasSuffix(src, ".gguf")
	switch {
	case gguf && opts.Quant == "q4k" && opts.StreamWeights:
		return src, nil
	case gguf && (opts.StreamWeights || DefaultToSidecar(false)):
		p, err := ensure()
		if err != nil {
			return "", fmt.Errorf("decoder: sidecar .giw for %s: %w — set Options.DirectLoad to load this .gguf straight into the heap instead", src, err)
		}
		return p, nil
	case DirSidecarApplies(src, opts) && DefaultToSidecar(false):
		p, err := ensure()
		if err != nil {
			logs.Warnf("decoder: loading %s directly: its sidecar .giw could not be built (%v)", src, err)
			return src, nil
		}
		return p, nil
	}
	return src, nil
}

// DirSidecarApplies reports whether a safetensors directory source takes the sidecar default: not with a LoRA to merge
// (the merge needs the safetensors base, and a plain sidecar would drop the adapter), not at q4k, bf16 or f16 (no .giw
// form), and not under StreamWeights, which streams only a .gguf's sidecar.
func DirSidecarApplies(src string, opts Options) bool {
	if opts.LoRA != "" || opts.Quant == "q4k" || opts.Quant == "bf16" || opts.Quant == "f16" || opts.StreamWeights || !sidecar.IsDir(src) {
		return false
	}
	m, _ := filepath.Glob(filepath.Join(src, "*.safetensors"))
	return len(m) > 0
}

// selfCheck verifies a bundle loads through the real mmap path: the weights deserialize and their shapes validate
// against the architecture. Whether the whole-payload CRC is recomputed is the load's own rule (decoder/giwverify.go):
// it is for a file with no marker, such as a cache found on disk or a bundle markWritten left unmarked, and it is not
// for a bundle the transcode has marked.
//
// Backend:"cpu", not Options{}: an EMPTY Backend means "needs canonical" (wantsCanonicalInt4's
// literal-"cpu"-is-a-promise rule), so Options{} would decline every kind-5 (row4-only) bundle this
// function itself wrote for a cpu-arm64 target. "cpu" accepts both kind 3 and kind 5 and matches how a
// cpu-arm64-target bundle is loaded in production.
//
// ResidentContext 1: the check runs no request, so it allocates no KV. Unpinned, the host fit guard
// prices the CPU's per-request ceiling over the model's whole window, prints a "context capped" line
// belonging to no real load, and under tight memory can refuse the check outright.
func selfCheck(path string) error {
	m, err := Load(path, Options{Backend: "cpu", ResidentContext: 1})
	if err != nil {
		return err
	}
	return m.Close()
}
