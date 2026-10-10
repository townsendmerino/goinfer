package decoder

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestForExperts pins the rule: with no sink the experts are built in order, one at a time; for a sink they are built
// across cores; and either way a failed expert's error is returned.
func TestForExperts(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	run := func(sink *giwWriter, fail int) (order []int, peak int, err error) {
		var mu sync.Mutex
		var inFlight, most atomic.Int64
		err = forExperts(12, sink, func(e int) error {
			cur := inFlight.Add(1)
			for {
				p := most.Load()
				if cur <= p || most.CompareAndSwap(p, cur) {
					break
				}
			}
			defer inFlight.Add(-1)
			time.Sleep(2 * time.Millisecond) // let the other workers start
			mu.Lock()
			order = append(order, e)
			mu.Unlock()
			if e == fail {
				return errors.New("planted")
			}
			return nil
		})
		return order, int(most.Load()), err
	}

	order, peak, err := run(nil, -1)
	if err != nil || peak != 1 || len(order) != 12 {
		t.Fatalf("no sink: err %v, %d in flight, %d built; want nil, 1, 12", err, peak, len(order))
	}
	for i, e := range order {
		if e != i {
			t.Fatalf("no sink: built in the order %v", order)
		}
	}
	if _, _, err := run(nil, 5); err == nil || err.Error() != "planted" {
		t.Errorf("no sink: a failed expert returned %v", err)
	}

	sink := &giwWriter{sink: io.Discard}
	// Four workers is a ceiling the scheduler may not fill on a busy box: over it is wrong, one is the bug.
	peak = 0
	for attempt := 0; attempt < 5 && peak < 2; attempt++ {
		order, peak, err = run(sink, -1)
		if err != nil || len(order) != 12 || peak > 4 {
			t.Fatalf("a sink: err %v, %d in flight, %d built; want nil, at most 4, 12", err, peak, len(order))
		}
	}
	if peak < 2 {
		t.Errorf("a sink: never more than %d expert in flight in five runs", peak)
	}
	if _, _, err := run(sink, 5); err == nil || err.Error() != "planted" {
		t.Errorf("a sink: a failed expert returned %v", err)
	}
}

// TestStreamTranscode_expertFanOutWritesTheSameBytes: every expert loop writes, with its experts built across cores,
// the bytes it writes with them built in order; a stream fans out and a resident load does not. The sources reach the
// per-expert tensors (two families' naming), the fused stack in its transposed layout, Nemotron's loop, Llama 4's
// whole-tensor read and the GGUF stacked experts. Gemma 4's two stacks and the fused stack in its row layout (Granite)
// have gitignored fixtures: they are checked where the fixture is on the box and reported as not checked elsewhere.
func TestStreamTranscode_expertFanOutWritesTheSameBytes(t *testing.T) {
	defer func() { expertsInOrderForTest = false }()
	for _, c := range []struct {
		src      string
		optional bool // a gitignored fixture: absent on a fresh clone
	}{
		{"../testdata/qwen3moe-tiny", false},
		{"testdata/qwen3_5_moe-tiny", false},
		{"../testdata/qwen3vlmoe-tiny", false},
		{"../testdata/nemotron3nano-tiny", false},
		{"../testdata/llama4-tiny", false},
		{"../testdata/glm-tiny.gguf", false},
		{"../testdata/gemma4-moe-tiny", true},
		{"../testdata/granite-tiny", true},
	} {
		if _, err := os.Stat(c.src); err != nil && c.optional {
			t.Logf("%s: not on this box, not checked", c.src)
			continue
		}
		stream := func() []byte {
			var b bytes.Buffer
			var err error
			if strings.HasSuffix(c.src, ".gguf") {
				_, err = StreamTranscodeGGUF(context.Background(), c.src, &b, "int4", false, GIWTargetNone, "t")
			} else {
				_, err = StreamTranscodeDir(context.Background(), c.src, &b, "int4", false, GIWTargetNone, "t")
			}
			if err != nil {
				t.Fatalf("%s: %v", c.src, err)
			}
			return b.Bytes()
		}

		expertsInOrderForTest = true
		before := expertFanoutsForTest.Load()
		ref := stream()
		if len(ref) < 1000 {
			t.Fatalf("%s: a %d-byte bundle: nothing was streamed", c.src, len(ref))
		}
		if n := expertFanoutsForTest.Load() - before; n != 0 {
			t.Fatalf("%s: %d fan-outs with the experts held in order", c.src, n)
		}

		expertsInOrderForTest = false
		got := stream()
		if n := expertFanoutsForTest.Load() - before; n == 0 {
			t.Errorf("%s: a stream built no experts across cores: the source does not reach its expert loop", c.src)
		}
		if !bytes.Equal(ref, got) {
			t.Errorf("%s: experts built across cores wrote %d bytes that differ from the in-order stream's %d", c.src, len(got), len(ref))
		}

		before = expertFanoutsForTest.Load()
		m, err := Load(c.src, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: resident load: %v", c.src, err)
		}
		m.Close()
		if n := expertFanoutsForTest.Load() - before; n != 0 {
			t.Errorf("%s: a resident load fanned its experts out %d times, inside layers that already run one per core", c.src, n)
		}
	}
}
