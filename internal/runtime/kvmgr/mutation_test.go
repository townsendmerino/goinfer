package kvmgr

// dropKeyPart removes one input from the key. It exists only in the test binary: nothing in a build can make a
// manager forget part of its key.
func (m *Manager) dropKeyPart(p keyPart) { m.parts &^= p }
