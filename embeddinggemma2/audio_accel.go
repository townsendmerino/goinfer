package embeddinggemma2

import (
	"fmt"
	"sort"

	"github.com/townsendmerino/aikit/audio"
)

// The audio tower on another device (Phase A's AM, docs/tasks/task-embeddinggemma2.md: Metal). An AudioAccelerator
// runs the 12 conformer blocks from aikit's loaded tower (its exported Layers); the subsampler before them and the
// output projection and embedder after them stay aikit's (Subsample, FinishBlocks), on the host.

// AudioAccelerator runs the Gemma 4 audio tower's blocks on another device.
type AudioAccelerator interface {
	Name() string
	// Blocks runs every conformer block over h [n, hidden] (the subsampler's output) and returns the last block's
	// output; h is not modified.
	Blocks(h []float32, n int) ([]float32, error)
	Close() error
}

var audAccels = map[string]func(*audio.Gemma4AudioEncoder) (AudioAccelerator, error){}

// RegisterAudioAccelerator makes an audio-tower accelerator available by name, from a backend module's init.
// UseAccelerator with the same name moves the audio tower too.
func RegisterAudioAccelerator(name string, factory func(*audio.Gemma4AudioEncoder) (AudioAccelerator, error)) {
	accelMu.Lock()
	defer accelMu.Unlock()
	audAccels[name] = factory
}

// AudioAccelerators lists the registered audio-tower accelerator names, sorted.
func AudioAccelerators() []string {
	accelMu.Lock()
	defer accelMu.Unlock()
	var n []string
	for k := range audAccels {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// NewAudioAccelerator builds the named audio-tower accelerator over a loaded tower.
func NewAudioAccelerator(name string, enc *audio.Gemma4AudioEncoder) (AudioAccelerator, error) {
	accelMu.Lock()
	f, ok := audAccels[name]
	accelMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("embeddinggemma2: no %q audio accelerator in this binary (registered: %v)", name, AudioAccelerators())
	}
	return f(enc)
}

// bindAudioAccel puts a loaded audio tower on the encoder's accelerator when one of the same name is registered; a
// decline leaves the CPU tower and is recorded in AudioDevice.
func (e *Encoder) bindAudioAccel() {
	a := e.aud
	if a == nil || e.accel == nil {
		return
	}
	if a.accel != nil {
		a.accel.Close()
		a.accel, a.device = nil, ""
	}
	name := e.accel.Name()
	accelMu.Lock()
	_, ok := audAccels[name]
	accelMu.Unlock()
	if !ok {
		a.device = "CPU (no " + name + " audio tower in this binary)"
		return
	}
	acc, err := NewAudioAccelerator(name, a.enc)
	if err != nil {
		a.device = "CPU (" + name + " declined: " + err.Error() + ")"
		return
	}
	a.accel, a.device = acc, name
}

// AudioAccelerator is the audio tower's accelerator, or nil when it runs on the CPU.
func (e *Encoder) AudioAccelerator() AudioAccelerator {
	if e.aud == nil {
		return nil
	}
	return e.aud.accel
}

// AudioFeaturesCPU is AudioFeaturesFrom on the CPU tower whatever the encoder's tower runs on (a test compares the
// two).
func (e *Encoder) AudioFeaturesCPU(mel []float32, T int) ([]float32, int, error) {
	if e.aud == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: audio is not enabled (EnableAudio)")
	}
	return e.audioFeaturesOn(nil, mel, T)
}

// audioFeaturesOn runs the tower with its blocks on a (nil: all on the CPU).
func (e *Encoder) audioFeaturesOn(a AudioAccelerator, mel []float32, T int) ([]float32, int, error) {
	enc := e.aud.enc
	if a == nil {
		f, err := enc.Forward(mel, T)
		if err != nil {
			return nil, 0, fmt.Errorf("embeddinggemma2: audio tower: %w", err)
		}
		return f, audio.Gemma4SoftTokens(T), nil
	}
	h, n, err := enc.Subsample(mel, T)
	if err == nil {
		h, err = a.Blocks(h, n)
	}
	var f []float32
	if err == nil {
		f, err = enc.FinishBlocks(h, n)
	}
	if err != nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: audio tower: %w", err)
	}
	return f, n, nil
}
