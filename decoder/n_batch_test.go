package decoder

import (
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"os"
	"strings"
	"sync"
	"testing"
)

// N-05: the GPTQ loader adds the v1 "+1" zero-point bias unconditionally and never read
// `checkpoint_format`. GPTQModel's v2 export drops that bias, so a v2 checkpoint dequantizes
// every weight ONE SCALE STEP LOW — no error, no NaN, a uniformly wrong model.
//
// Refused rather than implemented: there is no v2 checkpoint here to validate a v2 path
// against, and the failure being prevented is specifically the silent kind.
func TestParseQuantConfig_gptqCheckpointFormat(t *testing.T) {
	cfg := func(extra string) json.RawMessage {
		return json.RawMessage(`{"quant_method":"gptq","bits":4,"group_size":128` + extra + `}`)
	}
	for name, tc := range map[string]struct {
		extra   string
		wantErr bool
	}{
		"absent (older exports omit it)": {``, false},
		"explicit v1":                    {`,"checkpoint_format":"gptq"`, false},
		"gptq_v2 is refused":             {`,"checkpoint_format":"gptq_v2"`, true},
		"an unknown format is refused":   {`,"checkpoint_format":"something_new"`, true},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := parseQuantConfig(cfg(tc.extra))
			if tc.wantErr && err == nil {
				t.Error("accepted; every weight would load one scale step low, silently (N-05)")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("rejected a v1 checkpoint: %v", err)
			}
			if tc.wantErr && err != nil && !strings.Contains(err.Error(), "checkpoint_format") {
				t.Errorf("error %q does not name the field", err)
			}
		})
	}
	// AWQ shares this parser and has no such field — refusing it there would break a format
	// that never had the +1 problem.
	if _, err := parseQuantConfig(json.RawMessage(
		`{"quant_method":"awq","bits":4,"group_size":128,"checkpoint_format":"whatever"}`)); err != nil {
		t.Errorf("awq rejected for a gptq-only field: %v", err)
	}
}

// N-27: the router-capture buffers are package-level, were appended from inside the forward with
// NO LOCK, and had no bound. Under the documented concurrent-sequence contract two goroutines
// can be in a forward at once, so the appends raced on a slice header — a crash, not a wrong
// number — and on a long-running serve process the buffers grew without limit.
func TestRouterCapture_isBoundedAndRaceFree(t *testing.T) {
	prev := routerCapture
	routerCapture = true
	defer func() { routerCapture = prev; routerCaptureReset() }()
	routerCaptureReset()

	// Concurrent appends: under -race this is the assertion. Without the mutex it is a racing
	// slice append.
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 256 {
				routerCaptureDo(func() {
					routerCaptureBuf = append(routerCaptureBuf, []int{1, 2})
				})
			}
		}()
	}
	wg.Wait()
	if got := len(routerCaptureBuf); got != 8*256 {
		t.Errorf("captured %d entries, want %d — appends were lost", got, 8*256)
	}

	// The bound: past routerCaptureMax nothing more is appended, so a forgotten env var on a
	// serving process costs bounded memory rather than the process.
	routerCaptureReset()
	for range routerCaptureMax + 100 {
		routerCaptureDo(func() { routerCaptureBuf = append(routerCaptureBuf, []int{1}) })
	}
	if got := len(routerCaptureBuf); got != routerCaptureMax {
		t.Errorf("buffer holds %d entries, cap is %d — the bound does not hold (N-27)",
			got, routerCaptureMax)
	}

	// And capture OFF must stay a no-op, or the diagnostic costs something when unused.
	routerCapture = false
	routerCaptureReset()
	routerCaptureDo(func() { routerCaptureBuf = append(routerCaptureBuf, []int{9}) })
	if len(routerCaptureBuf) != 0 {
		t.Error("captured with the env unset")
	}
}

// N-07: the .giw reader allocated layer and expert structs BEFORE comparing the counts with the
// arch it had already resolved. validateShapes catches a mismatch, but only once every struct
// exists — and maxSerializedLayers is ~40x a real model, on an EXPORTED entry point.
//
// The check is now before the make(), so this asserts the ERROR arrives without the allocation
// happening. There is no way to observe "did not allocate" directly, so the assertion is that
// the failure is the ARCH-COUNT error rather than a later shape error: the arch comparison is
// the only thing that runs before the allocation, so getting that message proves the ordering.
func TestGiw_layerCountCheckedBeforeAllocation(t *testing.T) {
	const fixture = "../testdata/llama-tiny"
	m, err := Load(fixture, Options{})
	if err != nil {
		t.Skipf("no fixture at %s: %v", fixture, err)
	}
	blob, err := SerializeWeights(m.Weights(), "probe")
	m.Close()
	if err != nil {
		t.Fatalf("SerializeWeights: %v", err)
	}
	// Sanity: the untouched blob loads.
	if _, err := LoadSerializedWeights(blob); err != nil {
		t.Fatalf("the clean blob does not load: %v", err)
	}

	// Find the layer-count word by loading, then rewriting it to a huge value. It is the u32
	// immediately after the model-level tail, which we locate by searching for the real count
	// — crude, but this test only needs SOME blob whose declared count disagrees with the arch.
	w, _ := LoadSerializedWeights(blob)
	real32 := uint32(len(w.Layers))
	var idx = -1
	for i := 0; i+4 <= len(blob); i++ {
		if uint32(blob[i])|uint32(blob[i+1])<<8|uint32(blob[i+2])<<16|uint32(blob[i+3])<<24 == real32 {
			idx = i // first occurrence is fine; any u32 we corrupt must still be caught
			break
		}
	}
	if idx < 0 {
		t.Skipf("could not locate the layer-count word for %d layers", real32)
	}
	bad := append([]byte(nil), blob...)
	huge := uint32(maxSerializedLayers - 1) // in range for the old guard, wrong for the arch
	bad[idx], bad[idx+1] = byte(huge), byte(huge>>8)
	bad[idx+2], bad[idx+3] = byte(huge>>16), byte(huge>>24)

	// RESEAL THE CRC. Without this the blob fails the checksum first and the test passes
	// whether or not the arch check exists — measured: the first version of this test reported
	// success while proving only that CRC works. The CRC is over the body, appended last.
	binary.LittleEndian.PutUint32(bad[len(bad)-4:], crc32.ChecksumIEEE(bad[:len(bad)-4]))

	_, lerr := LoadSerializedWeights(bad)
	if lerr == nil {
		t.Fatal("a CRC-valid blob declaring a wrong layer count loaded cleanly (N-07)")
	}
	if !strings.Contains(lerr.Error(), "layer count") {
		t.Errorf("rejected with %q, want the arch layer-count error — anything else means the "+
			"check ran AFTER the allocation, which is what N-07 is about", lerr)
	}
}

// N-06: fp8Reconstruct's comment said the shape is checked against the ARCHITECTURE; the code
// compared only the element COUNT. out*in == in*out, so a transposed tensor passed and loaded
// with every weight in the wrong place.
//
// Unit-checked on the predicate, because building a valid fp8 safetensors file in-test is a
// large fixture for a one-line invariant: the property is that [out,in] and [in,out] must be
// distinguishable, which an element-count comparison cannot do.
func TestFP8_shapeCheckDistinguishesTranspose(t *testing.T) {
	const out, in = 4, 8
	countOnly := func(shape []int) bool { return shape[0]*shape[1] == out*in }
	shapeAware := func(shape []int) bool { return len(shape) == 2 && shape[0] == out && shape[1] == in }

	if !countOnly([]int{in, out}) {
		t.Fatal("premise broke: a transposed shape no longer has the same element count")
	}
	if shapeAware([]int{in, out}) {
		t.Error("the shape check accepts [in,out]; it must distinguish it from [out,in] (N-06)")
	}
	if !shapeAware([]int{out, in}) {
		t.Error("the shape check rejects the correct [out,in]")
	}
	// And the real function must use the shape-aware form.
	src, err := readSource("fp8.go")
	if err != nil {
		t.Fatalf("read fp8.go: %v", err)
	}
	if !strings.Contains(src, "wT.Shape[0] != out || wT.Shape[1] != in") {
		t.Error("fp8Reconstruct does not compare the declared shape against [out,in]: an " +
			"element-count check cannot tell a transposed tensor from a correct one (N-06)")
	}
}

// readSource reads a file from this package's directory.
func readSource(name string) (string, error) {
	b, err := os.ReadFile(name)
	return string(b), err
}
