//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// MEM13-copies-CUDA (docs/measurements/mem13-copies-cuda-2026-11/README.md): a Mamba-2 layer's in_proj and out_proj are planned while the build walks the
// layers and streamed to the device at upload, like every other dense projection (pack_stream.go), so the host holds no packed copy of them.
//
//	go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestMambaHostCopy_' -v ./backend/cuda/

// mambaLayersOf loads the nemotron-tiny fixture on CUDA and returns, for each Mamba-2 layer, what the build's walk makes of it.
func mambaLayersOf(t *testing.T) (m *decoder.Model, layers []int, p *mambaParams, hidden int) {
	t.Helper()
	var err error
	m, err = decoder.Load(filepath.Join("../../testdata", "nemotron-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load nemotron-tiny: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if p, err = buildMambaParams(m); err != nil || p == nil {
		t.Fatalf("nemotron-tiny has no Mamba-2 geometry: %v", err)
	}
	hidden, _, _, _, _, _, _ = m.Dims()
	for l, ly := range m.Program().Layers {
		if mix, _, ok := cudaMixerOf(ly); ok && mix == decoder.MixerMamba {
			layers = append(layers, l)
		}
	}
	if len(layers) == 0 {
		t.Fatal("nemotron-tiny has no Mamba-2 layer to read")
	}
	return m, layers, p, hidden
}

// G-1: packMambaLayer plans its two projections and holds no packed words; the buffers it names are the sizes packWeight would make.
func TestMambaHostCopy_planningHoldsNoPackedWords(t *testing.T) {
	m, layers, p, hidden := mambaLayersOf(t)
	for _, l := range layers {
		hm, err := packMambaLayer(m, l, hidden, p)
		if err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
		for name, h := range map[string]hostW{"in_proj": hm.in, "out_proj": hm.out} {
			if n := len(h.wpk) + len(h.ws) + len(h.ws16); n != 0 {
				t.Errorf("layer %d %s holds %d packed words before upload, want 0 (a planned weight is packed at upload)", l, name, n)
			}
			if h.src == nil {
				t.Errorf("layer %d %s is not a planned weight: it names no source for the upload to stream", l, name)
				continue
			}
			eager, err := packWeight(h.src)
			if err != nil {
				t.Fatalf("layer %d %s: packWeight: %v", l, name, err)
			}
			if h.kind != eager.kind || h.nWpk != len(eager.wpk) || h.nWs != len(eager.ws) || h.nWs16 != len(eager.ws16) {
				t.Errorf("layer %d %s: the plan names kind %s buffers %d/%d/%d, the packed weight is kind %s buffers %d/%d/%d", l, name,
					h.kind, h.nWpk, h.nWs, h.nWs16, eager.kind, len(eager.wpk), len(eager.ws), len(eager.ws16))
			}
		}
	}
}

// G-2: the device bytes a Mamba layer's projections get are the packed upload's, at one staging block and at several.
func TestMambaHostCopy_deviceBytesIdenticalToPackWeight(t *testing.T) {
	r := streamRig(t)
	m, layers, p, hidden := mambaLayersOf(t)
	for _, l := range layers {
		hm, err := packMambaLayer(m, l, hidden, p)
		if err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
		for name, h := range map[string]hostW{"in_proj": hm.in, "out_proj": hm.out} {
			if h.src == nil {
				continue // G-1 reports it
			}
			eager, err := packWeight(h.src)
			if err != nil {
				t.Fatalf("layer %d %s: packWeight: %v", l, name, err)
			}
			want := devBytes(t, r.upW(eager), h)
			for _, words := range []int{1 << 20, 64, 1} {
				withStage(t, words)
				r.pk = nil
				got := devBytes(t, r.upW(h), h)
				for k, b := range want {
					if string(got[k]) != string(b) {
						t.Errorf("layer %d %s, staging %d words: device buffer %s differs from the packed upload's", l, name, words, k)
					}
				}
			}
		}
	}
	if r.setupErr != nil {
		t.Fatalf("upload error: %v", r.setupErr)
	}
}

// anonSampler samples this process's anonymous resident bytes and swap every 50 ms, whole-process and inside one window (the build's: planStartForTest to the
// end of the load).
type anonSampler struct {
	mu                sync.Mutex
	peak, swapPeak    int64
	winStart, winPeak int64
	inWin             bool
	stop              chan struct{}
	done              chan struct{}
}

func readProcKB() (anon, swap int64) {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0, 0
	}
	for _, ln := range strings.Split(string(b), "\n") {
		var v int64
		switch {
		case strings.HasPrefix(ln, "RssAnon:"):
			fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(ln, "RssAnon:")), "%d", &v)
			anon = v
		case strings.HasPrefix(ln, "VmSwap:"):
			fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(ln, "VmSwap:")), "%d", &v)
			swap = v
		}
	}
	return
}

func startAnonSampler() *anonSampler {
	s := &anonSampler{stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(s.done)
		for {
			a, w := readProcKB()
			s.mu.Lock()
			s.peak, s.swapPeak = max(s.peak, a), max(s.swapPeak, w)
			if s.inWin {
				s.winPeak = max(s.winPeak, a)
			}
			s.mu.Unlock()
			select {
			case <-s.stop:
				return
			case <-time.After(50 * time.Millisecond):
			}
		}
	}()
	return s
}

func (s *anonSampler) beginWindow() {
	a, _ := readProcKB()
	s.mu.Lock()
	s.winStart, s.winPeak, s.inWin = a, a, true
	s.mu.Unlock()
}

func (s *anonSampler) finish() (peak, swap, winStart, winPeak int64) {
	a, _ := readProcKB()
	s.mu.Lock()
	if s.inWin {
		s.winPeak = max(s.winPeak, a)
	}
	s.inWin = false
	s.mu.Unlock()
	close(s.stop)
	<-s.done
	return s.peak, s.swapPeak, s.winStart, s.winPeak
}

// G-3 (tier 1): Nemotron-Nano-9B-v2 whole on CUDA, the cell TestNemotronResidentCUDA_tier1 calls "9b" (token table at int4, ctx 4096). It logs the load's whole-process
// peak anonymous resident bytes and peak swap, and the same inside the build's window: RssAnon when BuildResident begins planning its layers, its peak from there
// to the end of the load, and the growth between them (G-3b); then 16 greedy tokens. The README's A/B runs this on main and on the branch.
//
//	GOINFER_HEAVY_TESTS=1 go test -c -tags 'cuda goinfer_testhooks' -o /tmp/cuda.test ./backend/cuda/ && /tmp/cuda.test -test.run '^TestMambaHostCopy_tier1PeakAnon$' -test.v
func TestMambaHostCopy_tier1PeakAnon(t *testing.T) {
	requireHeavyModel(t)
	requireCUDADevice(t)
	path := os.ExpandEnv("$HOME/models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	smp := startAnonSampler()
	oldStart := planStartForTest
	planStartForTest = smp.beginWindow
	t.Cleanup(func() { planStartForTest = oldStart })
	mg, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", EmbedInt4: true, ResidentContext: 4096})
	if err != nil {
		smp.finish()
		t.Fatalf("load (cuda) %s: %v", path, err)
	}
	defer mg.Close()
	peak, swap, winStart, winPeak := smp.finish()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("VOID: the cuda resident declined %s (%s): %s", path, mg.DecodePath(), mg.ResidentDecline())
	}
	tokens := greedyReplyCUDA(t, rf, mg, g2rSeed(t, path)[:4], 16)
	t.Logf("TIER1 peak RssAnon %d MiB, peak VmSwap %d MiB; build window: RssAnon at plan start %d MiB, peak in window %d MiB, growth %d MiB; decode path %s, 16 greedy tokens %v",
		peak/1024, swap/1024, winStart/1024, winPeak/1024, (winPeak-winStart)/1024, mg.DecodePath(), tokens)
}
