package multimodal

import (
	"fmt"
	"sort"
	"sync"

	"github.com/townsendmerino/aikit/audio"
)

// The Whisper-style audio encoder on a device. Whisper's transcriber and Voxtral's tower both run aikit's audio.WhisperEncoder, so one registration moves both: a
// backend module registers a factory from its init, and a caller that loaded an encoder asks for the named device and hands it to WhisperEncoder.SetDevice.

var (
	whisperDevMu sync.Mutex
	whisperDevs  = map[string]func(*audio.WhisperEncoder) (audio.WhisperDevice, error){}
)

// RegisterWhisperDevice makes a device encoder available by name, from a backend module's init.
func RegisterWhisperDevice(name string, factory func(*audio.WhisperEncoder) (audio.WhisperDevice, error)) {
	whisperDevMu.Lock()
	defer whisperDevMu.Unlock()
	whisperDevs[name] = factory
}

// WhisperDevices lists the registered device names, sorted.
func WhisperDevices() []string {
	whisperDevMu.Lock()
	defer whisperDevMu.Unlock()
	var n []string
	for k := range whisperDevs {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// UseWhisperDevice builds the named device over a loaded encoder and sets it on enc. An error (no such registration, or the device declining: no memory, no
// kernel) leaves enc on the CPU, and the caller names the fallback.
func UseWhisperDevice(name string, enc *audio.WhisperEncoder) (audio.WhisperDevice, error) {
	whisperDevMu.Lock()
	f, ok := whisperDevs[name]
	whisperDevMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("multimodal: no %q Whisper encoder in this binary (registered: %v)", name, WhisperDevices())
	}
	d, err := f(enc)
	if err != nil {
		return nil, err
	}
	enc.SetDevice(d)
	return d, nil
}
