package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// streamDirSwapForTest writes layers 0 and 1 of a streamed directory transcode in each other's place: G-DS1's planted
// defect, which its byte-identity test must catch. Set only through SetStreamDirSwapForTest (goinfer_testhooks).
var streamDirSwapForTest bool

// StreamTranscodeDir is StreamTranscodeGGUF for a safetensors model DIRECTORY: it writes a .giw weights body (the GINFW
// serialization + trailing CRC) to out, loading ONE layer at a time and the embedding a row at a time, so peak memory is
// about the globals plus one layer rather than the whole model (docs/tasks/task-prequant-dir-streaming-2026-10.md).
// The bytes are those Load + SerializeWeightsToForTarget write for the same directory (G-DS1). Returns the bytes written.
//
// A family whose dedicated safetensors loader does not stream yet returns an error IsDirNoStream reports; the caller
// takes the resident transcode instead. int4mix and q4k are GGUF-only, as in Load.
func StreamTranscodeDir(ctx context.Context, dir string, out io.Writer, quant string, embedInt4 bool, target GIWTarget, id string) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	out = &ctxWriter{ctx: ctx, w: out} // cancellation observed at every write, as in StreamTranscodeGGUF (M-21)
	q, err := parseQuant(quant)
	if err != nil {
		return 0, err
	}
	if q == quantInt4Mix || q == quantQ4K {
		return 0, fmt.Errorf("decoder: %s is GGUF-only (got safetensors %s)", quant, dir)
	}
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		return 0, err
	}
	// Load writes the EOS ids generation_config.json adds back into the config a bundle serializes (decoder/model.go);
	// the stream writes the config in its head, before any layer, so it resolves them first (StreamTranscodeGGUF's M-04).
	if raw, jerr := json.Marshal(resolveEOSIDs(dir, cfg)); jerr == nil {
		cfg.EOSTokenID = raw
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		return 0, err
	}
	if serr := canSerialize(arch); serr != nil {
		return 0, serr
	}
	st, err := openCheckpointMmap(dir)
	if err != nil {
		return 0, err
	}
	defer st.Close()
	// needCanonical=true, skipRow4=false: the writer chooses what to emit from canonical bytes in RAM, per layer, exactly
	// as StreamTranscodeGGUF builds them.
	wr := &giwWriter{sink: out, target: target}
	if _, err := buildWeightsFromSafetensorsTo(cfg, arch, schema, st, q, embedInt4, true, false, nil, wr, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc)
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}
