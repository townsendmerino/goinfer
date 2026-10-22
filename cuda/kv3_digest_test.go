//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestKV3_prefillDigest prints one SHA-256 per prompt length over the last-token logits and every layer's K and V rows after
// PrefillLast on the real Qwen2.5-Coder-1.5B (int4, f16 KV), so two builds of the tree can be compared line by line: the identity
// gate of docs/measurements/kv3-cuda-2026-10-13.md §2.3 (b). It is self-contained, so it compiles on the tree before KV3 too.
//
//	GOINFER_HEAVY_TESTS=1 KV3_DIGEST=1 go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestKV3_prefillDigest$' -v ./cuda/ | grep KV3DIGEST
func TestKV3_prefillDigest(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("KV3_DIGEST") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and KV3_DIGEST=1")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFixtureFile(t, path)
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 13000, KVPrecision: "f16"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := mustResident(t, m)
	if v := os.Getenv("KV3_DIGEST_CHUNK"); v != "" { // a chunk wider than the prompt: the whole-prompt arm of the tail-rule exception
		decoder.SetKnobEnvForTest(t, m, "GOINFER_PREFILL_CHUNK", v)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	files, _ := filepath.Glob("../decoder/*.go")
	sort.Strings(files)
	var sb strings.Builder
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, _ := os.ReadFile(f)
		sb.Write(b)
		if sb.Len() > 300_000 {
			break
		}
	}
	ids, err := tk.Encode(sb.String(), false)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{300, 3900, 4100, 12000} {
		window := ids[1000 : 1000+n]
		e := make([][]float32, n)
		for i, id := range window {
			e[i] = append([]float32(nil), m.EmbedResidentForTest(id)...)
		}
		lg, err := r.PrefillLast(context.Background(), e, 0)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		h := sha256.New()
		var b4 [4]byte
		for _, v := range lg {
			binary.LittleEndian.PutUint32(b4[:], math.Float32bits(v))
			h.Write(b4[:])
		}
		err = r.do(func() error {
			if e := r.stream.Sync(); e != nil {
				return e
			}
			// The digest is over each layer's logical rows [0, n), read through the slot's table, so it does not depend on which pool
			// blocks hold them. The words are hashed in memory order: for an f16 layer that is the byte stream of its n*kvDim halves.
			for l := range r.layers {
				k, v, e := r.kvRowsForTest(r.kvSlot, l, 0, n)
				if e != nil {
					return e
				}
				for _, w := range [][]float32{k, v} {
					for _, x := range w {
						binary.LittleEndian.PutUint32(b4[:], math.Float32bits(x))
						h.Write(b4[:])
					}
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("KV3DIGEST n=%d sha256=%x\n", n, h.Sum(nil))
	}
}

func requireDeviceAndFixtureFile(t *testing.T, path string) {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
}
