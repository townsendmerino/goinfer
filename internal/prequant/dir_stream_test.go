package prequant

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// dirStreamCases are G-DS1's configurations (docs/tasks/task-prequant-dir-streaming-2026-10.md): int4 for the Metal
// target with the int4 embedding, int4 for the CPU target, and int8int8.
var dirStreamCases = []struct {
	quant     string
	embedInt4 bool
	target    decoder.GIWTarget
}{
	{"int4", true, decoder.GIWTargetForBackend("metal")},
	{"int4", false, decoder.GIWTargetForBackend("cpu")},
	{"int8int8", false, decoder.GIWTargetForBackend("cpu")},
}

// dirStreamCompare transcodes dir both ways and returns the resident and streamed bodies; skip is non-empty when the
// directory is not a model the resident path builds, noStream when its family's loader does not stream.
func dirStreamCompare(t *testing.T, dir, quant string, embedInt4 bool, target decoder.GIWTarget) (res, str []byte, skip string, noStream bool) {
	t.Helper()
	var rb, sb bytes.Buffer
	id := filepath.Base(dir)
	if _, err := residentDirBody(&rb, dir, "", quant, embedInt4, target, id); err != nil {
		return nil, nil, err.Error(), false
	}
	if _, err := decoder.StreamTranscodeDir(context.Background(), dir, &sb, quant, embedInt4, target, id); err != nil {
		if decoder.IsDirNoStream(err) {
			return rb.Bytes(), nil, "", true
		}
		t.Fatalf("streamed transcode failed where the resident one did not: %v", err)
	}
	return rb.Bytes(), sb.Bytes(), "", false
}

// dirStreamDiff compares a resident and a streamed bundle as G-DS1 is amended (2026-10-09, the task doc): byte-identical
// before the v5 quant label and after its alignment pad, both CRCs valid. The streamed bundle records the label as ""
// because the header is written before any layer exists (B11, exactly as StreamTranscodeGGUF's bundles); a reader infers
// it from the identical tensors. giwSplit checks the pad is zeros and the stored CRC matches each body.
func dirStreamDiff(t *testing.T, res, str []byte) string {
	t.Helper()
	at := giwLabelOffset(t, res)
	n := int(binary.LittleEndian.Uint32(res[at : at+4]))
	rPre, rPost := giwSplit(t, res, at, string(res[at+4:at+4+n]))
	sPre, sPost := giwSplit(t, str, giwLabelOffset(t, str), "")
	if !bytes.Equal(rPre, sPre) {
		return fmt.Sprintf("the bundles differ before the quant label (%d vs %d B)", len(rPre), len(sPre))
	}
	if !bytes.Equal(rPost, sPost) {
		at := min(len(rPost), len(sPost))
		for i := range at {
			if rPost[i] != sPost[i] {
				at = i
				break
			}
		}
		return fmt.Sprintf("the bundles differ after the quant label (%d vs %d B, first difference %d bytes in)", len(rPost), len(sPost), at)
	}
	return ""
}

// TestDirStream_byteIdentical (G-DS1): for every safetensors fixture the resident transcode builds, the streamed bundle body
// is byte-identical to it, in each configuration. Families whose loader does not stream are listed, not compared (their
// bundle comes from the resident path).
func TestDirStream_byteIdentical(t *testing.T) {
	ents, err := os.ReadDir("../../testdata")
	if err != nil {
		t.Fatal(err)
	}
	var same, noStream []string
	for _, e := range ents {
		dir := filepath.Join("../../testdata", e.Name())
		if !e.IsDir() || !hasSafetensors(dir) {
			continue
		}
		for _, c := range dirStreamCases {
			res, str, skip, ns := dirStreamCompare(t, dir, c.quant, c.embedInt4, c.target)
			label := e.Name() + "/" + c.quant
			switch {
			case skip != "":
				continue
			case ns:
				noStream = append(noStream, label)
				continue
			}
			if why := dirStreamDiff(t, res, str); why != "" {
				t.Errorf("%s: %s", label, why)
				continue
			}
			same = append(same, label)
		}
	}
	t.Logf("byte-identical: %d (%s)", len(same), strings.Join(same, " "))
	t.Logf("not streamed (resident path): %d (%s)", len(noStream), strings.Join(noStream, " "))
	if len(same) < 20 {
		t.Errorf("only %d configurations compared: the fixture set no longer covers what this gate was registered on", len(same))
	}
}
