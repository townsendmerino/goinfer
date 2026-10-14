//go:build cuda

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// i3CodexMellum is every /v1/responses request of the I3 night's Codex x Mellum2.1 cell (serve --backend cuda --ctx 16384, window 1024), in
// order: the run it belongs to, prompt tokens, completion tokens, and the server's time to first token in ms. Source and method:
// docs/measurements/prefix-reuse-windowed-kv-2026-10-14.md.
var i3CodexMellum = [][4]int{
	{0, 6116, 319, 61315}, {0, 6208, 258, 62296}, {0, 6292, 413, 914}, {1, 6102, 248, 61365}, {1, 6204, 92, 62250}, {1, 6492, 253, 2902},
	{1, 6836, 225, 69073}, {1, 6938, 112, 1136}, {1, 7154, 374, 2196}, {2, 6105, 625, 61042}, {2, 6211, 545, 62378}, {2, 6353, 167, 64051},
	{2, 6580, 214, 2312}, {2, 6741, 104, 68078}, {2, 7032, 136, 2973}, {2, 7200, 317, 1800}, {2, 7304, 125, 1145}, {2, 7593, 643, 2985},
	{2, 7700, 73, 77585}, {2, 7805, 74, 1168}, {2, 8006, 92, 2160}, {2, 8110, 104, 1194}, {2, 8214, 374, 85598}, {2, 8445, 247, 2404},
	{2, 8751, 3933, 3091}, {2, 7272, 1887, 73313}, {2, 6543, 630, 65962}, {2, 6638, 309, 66768}, {2, 6866, 584, 69571}, {2, 7140, 436, 71959},
	{2, 7429, 4096, 74985}, {3, 6108, 280, 61248}, {3, 6199, 125, 62418}, {3, 6668, 151, 4593}, {3, 6854, 297, 69371}, {3, 7090, 582, 2439},
	{3, 7179, 31, 72365}, {3, 7362, 446, 1722}, {3, 7408, 567, 75151}, {3, 7498, 724, 75740}, {3, 7591, 95, 76705}, {3, 7774, 228, 78832},
	{3, 7911, 1328, 1494}, {3, 7957, 1398, 80737}, {3, 8094, 1009, 82380}, {3, 8372, 111, 84694}, {3, 8508, 121, 1530}, {3, 8764, 2086, 2610},
	{3, 9039, 68, 91760}, {3, 9300, 248, 2760}, {3, 9526, 91, 2344}, {3, 9623, 334, 97968}, {4, 6109, 230, 61295}, {4, 6200, 239, 62459},
	{4, 6567, 150, 3627}, {4, 6801, 82, 68491}, {4, 7046, 858, 2523}, {4, 7138, 841, 71991}, {4, 7230, 651, 72984},
}

type replayed struct {
	cold, runStart bool
	rollback       int // positions the slot held past the reused prefix
}

// replayWindowedKV drives kvWindowNextBase and kvWindowReusable over the recorded requests the way the CUDA resident saw them: one slot, a
// prompt prefilled in passes of min(512, slack+1) rows from the reused prefix, then one write per generated token but the last. Codex sends
// no reasoning back, so a follow-up inside a run diverges at the previous prompt's end. A run's first request shares the instructions and tools
// with the last run; its divergence point was not recorded and is taken as 30 tokens before its end.
func replayWindowedKV(t *testing.T, slack, rollback int) []replayed {
	t.Helper()
	const window = 1024
	capN, chunk := window+slack, min(prefillDefaultChunk, slack+1)
	base, held := 0, 0
	var out []replayed
	write := func(p0, n int) {
		_, to, err := kvWindowNextBase(base, p0, n, window, capN, rollback)
		if err != nil {
			t.Fatalf("slack %d rollback %d: %v", slack, rollback, err)
		}
		base = to
	}
	for i, rq := range i3CodexMellum {
		run, prompt, gen := rq[0], rq[1], rq[2]
		first := i == 0 || i3CodexMellum[i-1][0] != run
		lcp := prompt - 30
		if !first {
			lcp = min(i3CodexMellum[i-1][1], prompt-1)
		}
		lcp = min(lcp, held)
		reuse := lcp
		if !kvWindowReusable(base, lcp, window) {
			reuse = 0
		}
		out = append(out, replayed{cold: reuse == 0, runStart: first, rollback: held - lcp})
		for p := reuse; p < prompt; p += chunk {
			write(p, min(chunk, prompt-p))
		}
		for p := prompt; p < prompt+gen-1; p++ {
			write(p, 1)
		}
		held = prompt + gen - 1
	}
	return out
}

// TestWindowedKV_i3Replay replays the I3 Codex x Mellum2.1 traffic through the windowed-KV arithmetic. With no rollback reserve (slack 512,
// rollback 0) it must reproduce the cold prefills the server logged, request for request: that is the cause. With the production constants
// every in-run follow-up whose rollback is within decoder.WindowedKVRollback must reuse its prefix.
func TestWindowedKV_i3Replay(t *testing.T) {
	old := replayWindowedKV(t, 512, 0)
	nCold := 0
	for i, r := range old {
		observed := i3CodexMellum[i][3] > 20000 // a cold 6-10K prefill takes 61-98 s; a warm one, 0.9-4.6 s
		if r.cold != observed {
			t.Errorf("request %d (prompt %d): replay says cold=%v, the server's ttft of %d ms says cold=%v", i, i3CodexMellum[i][1], r.cold, i3CodexMellum[i][3], observed)
		}
		if r.cold {
			nCold++
		}
	}
	if nCold != 35 {
		t.Errorf("old rule: %d cold prefills, the server logged 35", nCold)
	}
	now := replayWindowedKV(t, decoder.WindowedKVSlack, decoder.WindowedKVRollback)
	inRun, inRunCold := 0, 0
	for i, r := range now {
		if r.runStart {
			continue
		}
		inRun++
		if r.cold {
			inRunCold++
			if r.rollback <= decoder.WindowedKVRollback {
				t.Errorf("request %d (prompt %d): cold after a rollback of %d positions, within the %d the slot keeps", i, i3CodexMellum[i][1], r.rollback, decoder.WindowedKVRollback)
			}
		}
	}
	t.Logf("old rule: %d of %d cold; now: %d of %d in-run follow-ups cold (each rolls back more than %d)", nCold, len(old), inRunCold, inRun, decoder.WindowedKVRollback)
}
