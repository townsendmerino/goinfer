//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// mem6EModelLong is gemma4-emodel-tiny with its position window raised to 4,096 (the weights do not depend on it), so a prompt
// spans several of the routes' 256-row chunks.
func mem6EModelLong(t *testing.T) string {
	t.Helper()
	dst := t.TempDir()
	ents, err := os.ReadDir(eModelDir)
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(eModelDir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if e.Name() == "config.json" {
			var c map[string]any
			if err := json.Unmarshal(b, &c); err != nil {
				t.Fatal(err)
			}
			for _, m := range []map[string]any{c, asMap(c["text_config"])} {
				if m != nil {
					if _, ok := m["max_position_embeddings"]; ok {
						m["max_position_embeddings"] = 4096
					}
				}
			}
			if b, err = json.Marshal(c); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

func asMap(v any) map[string]any { m, _ := v.(map[string]any); return m }

// mem6MainHashes are main's last-logits-plus-32-steps hashes for TestMEM6_prefillHostBytes's prefills (MEM6-G2), by route and
// prompt rows, taken on main before the change (docs/tasks/task-memory-campaign-2026-10.md, "MEM6 (Metal routes) gates").
var mem6MainHashes = map[string]map[int]uint64{
	"batched":     {1024: 0xd183d8b68b01a1ab, 2048: 0xa71ddffe5ef5e90},
	"layer-major": {1024: 0x88f6214a681043ac, 2048: 0xc40ab360a5668f7c},
}

// TestMEM6_prefillHostBytes is MEM6-G1 and MEM6-G2 (docs/tasks/task-memory-campaign-2026-10.md, "MEM6 (Metal routes) gates"): on
// the E-model fixture, Metal-resident, on the batched route and the layer-major one, a resident prefill of 1,024 and of 2,048 rows:
// the most embedding rows the decoder holds at once (at most 256), and on the batched route the Go bytes a prompt row adds (under
// an eighth of an embedding row); and the last logits and 32 decode steps after them, hashed, equal to main's.
func TestMEM6_prefillHostBytes(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := mem6EModelLong(t)
	for _, batched := range []bool{true, false} {
		route := map[bool]string{true: "batched", false: "layer-major"}[batched]
		t.Run(route, func(t *testing.T) {
			defer func(v bool) { emodelBatchedOn = v }(emodelBatchedOn)
			emodelBatchedOn = batched
			m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 2112})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok || !a.emodelLayerMajor() {
				t.Fatalf("not the E-model resident (%v): %s", ok, m.ResidentDecline())
			}
			_, _, _, _, _, _, vocab := m.Dims()
			held := 0
			decoder.SetEmbedRowsObserverForTest(func(rows int) { held = max(held, rows) })
			defer decoder.SetEmbedRowsObserverForTest(nil)
			bytesFor := func(n int) (uint64, uint64) {
				ids := make([]int, n)
				for i := range ids {
					ids[i] = (i*37 + 11) % vocab
				}
				runtime.GC()
				var s0, s1 runtime.MemStats
				runtime.ReadMemStats(&s0)
				lg, err := m.ResidentPrefillSeedForTest(context.Background(), ids)
				runtime.ReadMemStats(&s1)
				if err != nil {
					t.Fatal(err)
				}
				h := fnv.New64a()
				put := func(l []float32) {
					for _, x := range l {
						b := math.Float32bits(x)
						h.Write([]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)})
					}
				}
				put(lg)
				next := argmaxF(lg)
				for j := range 32 {
					l, err := a.Forward(m.EmbedResidentForTest(next), n+j)
					if err != nil {
						t.Fatal(err)
					}
					put(l)
					next = argmaxF(l)
				}
				return s1.TotalAlloc - s0.TotalAlloc, h.Sum64()
			}
			b1, h1 := bytesFor(1024)
			b2, h2 := bytesFor(2048)
			perRow := float64(int64(b2)-int64(b1)) / 1024
			embRow := float64(4 * m.ResidentEmbedLen())
			fmt.Fprintf(os.Stderr, "[MEM6] %s: at most %d embedding rows held at once; Go bytes allocated by the prefill: %d at 1,024 rows, %d at 2,048; %.0f B per prompt row against %.0f B per embedding row (%.3f); logits+decode hash %#x (1,024) %#x (2,048)\n",
				route, held, b1, b2, perRow, embRow, perRow/embRow, h1, h2)
			if held > 256 {
				t.Errorf("%s: the decoder held %d embedding rows at once, over one 256-row chunk", route, held)
			}
			if batched && perRow >= embRow/8 {
				t.Errorf("%s: a prompt row adds %.0f B, at least an eighth of an embedding row (%.0f B): still O(prompt)", route, perRow, embRow)
			}
			if h1 != mem6MainHashes[route][1024] || h2 != mem6MainHashes[route][2048] {
				t.Errorf("%s: logits hashes %#x, %#x differ from main's %#x, %#x", route, h1, h2, mem6MainHashes[route][1024], mem6MainHashes[route][2048])
			}
		})
	}
}
