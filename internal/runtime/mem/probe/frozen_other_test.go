//go:build !linux && !darwin && !windows

package probe

func frozenReaders() frozen { return frozen{} }
