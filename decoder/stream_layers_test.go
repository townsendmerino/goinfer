package decoder

import (
	"bytes"
	"context"
	"errors"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// streamLayersProbe drives streamLayers over n fake layers and reports the order layers were built in and the most that
// were in flight at once. Every layer writes the same number of bytes.
func streamLayersProbe(t *testing.T, n int, arch *Architecture, budget func(layerBytes int64) int64, fail int) (order []int, maxInFlight int, err error) {
	t.Helper()
	mk := func(l *LayerWeights) { l.QProj = linalg.WrapF32(make([]float32, 64), 8, 8) }
	probe := &giwWriter{sink: io.Discard, arch: arch}
	var one LayerWeights
	mk(&one)
	probe.layer(&one)
	w := &Weights{arch: arch, Layers: make([]LayerWeights, n)}
	sink := &giwWriter{sink: io.Discard, arch: arch, layerBudget: budget(probe.n)}
	var mu sync.Mutex
	var inFlight, peak atomic.Int64
	err = streamLayers(w, arch, sink, func(i int) error {
		cur := inFlight.Add(1)
		for {
			p := peak.Load()
			if cur <= p || peak.CompareAndSwap(p, cur) {
				break
			}
		}
		defer inFlight.Add(-1)
		time.Sleep(2 * time.Millisecond) // let the rest of the window start
		mu.Lock()
		order = append(order, i)
		mu.Unlock()
		if i == fail {
			return errors.New("planted")
		}
		mk(&w.Layers[i])
		return nil
	})
	return order, int(peak.Load()), err
}

// TestStreamLayers_window pins the window rule: layer 0 is built alone; a MoE never builds two layers at once; a dense
// model's window is the budget over a layer's bytes, capped by GOMAXPROCS and floored at one; a writer with no budget
// builds one at a time; and a failed load is returned.
func TestStreamLayers_window(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	dense, moe := &Architecture{}, &Architecture{MoE: &MoEConfig{NumExperts: 2, TopK: 1}}
	for _, c := range []struct {
		name   string
		arch   *Architecture
		budget func(int64) int64
		want   int // the most layers in flight
	}{
		{"dense, a budget of many layers: GOMAXPROCS caps it", dense, func(b int64) int64 { return 100 * b }, 4},
		{"dense, a budget of two and a half layers", dense, func(b int64) int64 { return 2*b + b/2 }, 2},
		{"dense, a budget under one layer", dense, func(b int64) int64 { return b / 2 }, 1},
		{"dense, no budget set", dense, func(int64) int64 { return 0 }, 1},
		{"MoE, a budget of many layers", moe, func(b int64) int64 { return 100 * b }, 1},
	} {
		// The window is a ceiling the scheduler may not fill on a busy box: over it is always wrong, under it gets retried.
		peak := 0
		for attempt := 0; attempt < 5 && peak != c.want; attempt++ {
			var order []int
			var err error
			order, peak, err = streamLayersProbe(t, 13, c.arch, c.budget, -1)
			if err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			if peak > c.want {
				t.Fatalf("%s: %d layers in flight, over the window of %d", c.name, peak, c.want)
			}
			if len(order) != 13 || order[0] != 0 {
				t.Fatalf("%s: built %v, want all 13 with layer 0 first and alone", c.name, order)
			}
		}
		if peak != c.want {
			t.Errorf("%s: %d layer(s) in flight at most in five attempts, want %d", c.name, peak, c.want)
		}
	}
	if _, _, err := streamLayersProbe(t, 13, dense, func(b int64) int64 { return 100 * b }, 6); err == nil || err.Error() != "planted" {
		t.Errorf("a failed layer load returned %v, want the load's own error", err)
	}
}

// TestStreamLayers_everyWindowWritesTheSameBytes: a dense model streamed with windows of 2, 3 and more layers than it has
// writes the bytes a one-layer-at-a-time stream writes, from a safetensors directory and from a GGUF.
func TestStreamLayers_everyWindowWritesTheSameBytes(t *testing.T) {
	defer func() { streamLayerWindowForTest = 0 }()
	for _, c := range []struct {
		src    string
		stream func(w io.Writer) error
	}{
		{"../testdata/olmo3-tiny", func(w io.Writer) error {
			_, err := StreamTranscodeDir(context.Background(), "../testdata/olmo3-tiny", w, "int4", true, GIWTargetNone, "t")
			return err
		}},
		{"../testdata/gemma2-hd-tiny.gguf", func(w io.Writer) error {
			_, err := StreamTranscodeGGUF(context.Background(), "../testdata/gemma2-hd-tiny.gguf", w, "int4", false, GIWTargetNone, "t")
			return err
		}},
	} {
		var ref []byte
		for _, k := range []int{1, 2, 3, 64} {
			streamLayerWindowForTest = k
			var b bytes.Buffer
			if err := c.stream(&b); err != nil {
				t.Fatalf("%s, window %d: %v", c.src, k, err)
			}
			if k == 1 {
				ref = b.Bytes()
				if len(ref) < 1000 {
					t.Fatalf("%s: a %d-byte bundle: nothing was streamed", c.src, len(ref))
				}
				continue
			}
			if !bytes.Equal(ref, b.Bytes()) {
				t.Errorf("%s: a window of %d wrote %d bytes that differ from the one-layer stream's %d", c.src, k, b.Len(), len(ref))
			}
		}
	}
}
