package gpu

// Releasing a resident ModelW.
//
// Context.Close() releases the DEVICE, but not the buffers a caller uploaded through
// UploadF32/UploadW8A8 — those are caller-owned, and each live one holds a reference that
// keeps the device's memory alive. DecodeRunner.Release says so explicitly for the scratch
// ("frees the runner's scratch, not the resident model"); nothing said it for the model.
//
// The consequence was a test-suite failure that read as something else entirely: a full
// ./gpu/ run climbed to 7,782 MiB of 8,192 and then every later test failed with
// "gpu: request device: failed to request device" — an out-of-memory wearing an
// unrelated error message. Because the tests that lost their device then SKIP rather than
// fail, TestWebGPU_forwardParity and TestResidentForwardN_parity quietly stopped being gates.
//
// Measured while diagnosing it, and worth recording because two plausible causes were
// refuted before the real one: 200 Context create/destroy cycles in one process are fine
// (churn is not the problem), 63 Contexts can be LIVE at once (a real limit, but the counter
// read ~0 at the failure), and file descriptors plateau at 128 against a 524,288 limit.
// The only thing that actually ran out was VRAM.

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
	a.Norm, a.InvFreq, a.KCache, a.VCache = nil, nil, nil, nil
	a.QProj, a.KProj, a.VProj, a.OProj = nil, nil, nil, nil
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
