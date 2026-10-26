//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/audio"
)

// TestAudioTowerClose_leavesTheOtherModelsServing: closing Voxtral's device tower, as serve's unload does, releases that tower
// only. A Whisper tower and a resident decoder on the same card go on producing exactly what they produced before it. Each
// tower makes its own Device (newTowerOps), whose release frees its own allocations and drops one retain of the primary
// context (cuDevicePrimaryCtxRelease, never a reset); the resident holds its own Device.
func TestAudioTowerClose_leavesTheOtherModelsServing(t *testing.T) {
	vox, err := audio.LoadVoxtralAudio("../testdata/voxtral-tiny")
	if err != nil {
		t.Skipf("no tiny Voxtral fixture: %v", err)
	}
	defer vox.Close()
	wsp, err := audio.LoadWhisperEncoder("../testdata/whisper-tiny-rand")
	if err != nil {
		t.Skipf("no tiny Whisper fixture: %v", err)
	}
	defer wsp.Close()
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("no CUDA load: %v", err)
	}
	defer m.Close()
	if m.ResidentForwardForTest() == nil {
		t.Skip("the decoder did not go resident on this device")
	}
	generate := func() []int {
		out, g := m.Generate(context.Background(), []int{1, 2, 3}, 8, decoder.SamplingParams{})
		var ids []int
		for id := range out {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil || len(ids) == 0 {
			t.Fatalf("resident generation: %d tokens, err %v", len(ids), err)
		}
		return ids
	}
	before := generate()

	tv, err := newAudioTower(vox.Enc)
	if err != nil {
		t.Skipf("no CUDA device for the audio towers: %v", err)
	}
	tw, err := newAudioTower(wsp)
	if err != nil {
		t.Fatal(err)
	}
	defer tw.Close()
	if _, err := tv.Forward(audioRandFeats(vox.Enc.Cfg.MelBins, 5)); err != nil {
		t.Fatal(err)
	}
	fw := audioRandFeats(wsp.Cfg.MelBins, 6)
	want, err := tw.Forward(fw)
	if err != nil {
		t.Fatal(err)
	}

	if err := tv.Close(); err != nil {
		t.Fatalf("closing Voxtral's tower: %v", err)
	}
	got, err := tw.Forward(fw)
	if err != nil || !slices.Equal(got, want) {
		t.Errorf("the Whisper tower after Voxtral's closed: err %v, output identical %v", err, slices.Equal(got, want))
	}
	if after := generate(); !slices.Equal(after, before) {
		t.Errorf("the resident decoder after Voxtral's tower closed: %v, before %v", after, before)
	}
}
