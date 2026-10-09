package prequant

import (
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// projectedSidecarBytes is the size a sidecar built at quant will have, projected from the source
// GGUF's tensor shapes (a header read — no weights are paged in). The pre-transcode disk check used
// the SOURCE's size as its proxy, on the argument that a sidecar is never bigger than an f32 source;
// but the sources are already quantized, and real int4 sidecars measured 1.02–1.16× their q4_k_m
// source, int8int8 about 1.6× — so the check passed and the transcode could still run out of disk.
//
// Pricing, per element, the payload plus its scales, rounded up:
//   - matrices at the quant: int4 0.5 B + an f32 scale per group of 32 (per row, rounded up);
//     int8/int8int8 1 B + scales;
//     int4mix puts FFN tensors at int4 and the rest at int8; f32 4 B;
//   - the token embedding and LM head at int8 under int4 and int4mix (the loader's pin);
//   - vectors (norms, biases) at f32.
//
// A fixed allowance per tensor and per file covers the header, shapes and alignment, and a 3% margin the
// rest (on a tiny model those fixed costs are most of the file; on a real one, noise). ok is false when the header does not
// parse; the caller then falls back to the source's size.
func projectedSidecarBytes(ggufPath, quant string) (int64, bool) {
	g, err := decoder.OpenGGUFMmap(ggufPath)
	if err != nil {
		return 0, false
	}
	defer g.Close()
	const (
		perTensor = 256     // name, kind, shape and alignment padding
		perFile   = 1 << 16 // header, config and the frame
	)
	total := float64(perFile)
	for _, name := range g.Names() {
		dims, ok := g.Dims(name)
		if !ok {
			return 0, false
		}
		total += perTensor
		n := 1.0
		for _, d := range dims {
			n *= float64(d)
		}
		if len(dims) < 2 {
			total += n * 4 // vectors stay f32
			continue
		}
		// Scales are per group of 32 along a row (GGUF dims[0] is the row), rounded up per row.
		cols := float64(dims[0])
		rows := n / cols
		groups := rows * float64((dims[0]+31)/32)
		int4 := n*0.5 + groups*4
		int8 := n + groups*4
		pinned := strings.HasPrefix(name, "token_embd") || strings.HasPrefix(name, "output.")
		switch quant {
		case "", "f32":
			total += n * 4
		case "int4":
			if pinned {
				total += int8
			} else {
				total += int4
			}
		case "int4mix":
			if !pinned && strings.Contains(name, "ffn_") {
				total += int4
			} else {
				total += int8
			}
		default: // int8, int8int8
			total += int8
		}
	}
	return int64(total * 1.03), true
}

// projectedDirSidecarBytes is projectedSidecarBytes for a safetensors directory (S18): the sizes of its .safetensors
// files, scaled from a 16-bit source to the quant's bytes per element with its group scales (int4 0.5 + 4/32, int8 1 +
// 4/32, f32 4), plus the same 3% margin. A bundled vision tower is in the files and not in the sidecar, and an f32 source
// is twice the elements' bytes, so both over-count, which is the safe direction for a disk check. ok is false when the
// directory holds no .safetensors file.
func projectedDirSidecarBytes(dir, quant string) (int64, bool) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return 0, false
	}
	var src int64
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		if fi, err := e.Info(); err == nil {
			src += fi.Size()
		}
	}
	if src == 0 {
		return 0, false
	}
	perElem := 1 + 4.0/32 // int8, int8int8, int4mix (its attention stays int8)
	switch quant {
	case "", "f32":
		perElem = 4
	case "int4":
		perElem = 0.5 + 4.0/32
	}
	return int64(float64(src) / 2 * perElem * 1.03), true
}
