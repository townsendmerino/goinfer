//go:build gpu

package gpu

// Releasing a resident ModelW.
//
// Context.Close() releases the DEVICE, but not the buffers a caller uploaded through UploadF32/UploadW8A8: those are
// caller-owned, and each live one holds a reference that keeps the device's memory alive. DecodeRunner.Release frees only the
// runner's scratch, not the resident model, so a ModelW must be released with ModelW.Release. A leak here ends as "failed to
// request device" in later tests (an out-of-memory wearing an unrelated error), and because tests that lose their device SKIP
// rather than fail, the parity gates quietly stop being gates (TestNoBufferLeak, bufaccount.go).
// History: docs/code-notes/gpu.md#Releasing a resident ModelW.

// Release frees every device buffer this resident model owns and zeroes the handles, so a
// double Release is safe and a use-after-free is a nil dereference at the Go boundary rather
// than undefined behaviour inside the native layer.
func (m *ModelW) Release() {
	if m == nil {
		return
	}
	for i := range m.Layers {
		m.Layers[i].Release()
	}
	m.Layers = nil
	closeDeviceBuffer(m.FinalNorm)
	closeResidentW8A8(m.LMHead)
	m.FinalNorm, m.LMHead = nil, nil
}

// Release frees one layer's resident weights.
func (l *LayerW) Release() {
	if l == nil {
		return
	}
	l.Attn.Release()
	closeDeviceBuffer(l.MLPNorm)
	closeResidentW8A8(l.Gate)
	closeResidentW8A8(l.Up)
	closeResidentW8A8(l.Down)
	l.MLPNorm, l.Gate, l.Up, l.Down = nil, nil, nil, nil
}

// Release frees one attention block's resident weights, including its KV cache — which is
// the largest single allocation in a deep-context model and the one most worth reclaiming.
func (a *AttnWeights) Release() {
	if a == nil {
		return
	}
	closeDeviceBuffer(a.Norm)
	closeResidentW8A8(a.QProj)
	closeResidentW8A8(a.KProj)
	closeResidentW8A8(a.VProj)
	closeResidentW8A8(a.OProj)
	closeDeviceBuffer(a.InvFreq)
	closeDeviceBuffer(a.KCache)
	closeDeviceBuffer(a.VCache)
	closeDeviceBuffer(a.QBias)
	closeDeviceBuffer(a.KBias)
	closeDeviceBuffer(a.VBias)
	closeDeviceBuffer(a.QNorm)
	closeDeviceBuffer(a.KNorm)
	a.Norm, a.InvFreq, a.KCache, a.VCache = nil, nil, nil, nil
	a.QProj, a.KProj, a.VProj, a.OProj = nil, nil, nil, nil
	a.QBias, a.KBias, a.VBias = nil, nil, nil
	a.QNorm, a.KNorm = nil, nil
}

func closeDeviceBuffer(d *DeviceBuffer) {
	if d != nil {
		_ = d.Close()
	}
}

func closeResidentW8A8(r *ResidentW8A8) {
	if r != nil {
		_ = r.Close()
	}
}
