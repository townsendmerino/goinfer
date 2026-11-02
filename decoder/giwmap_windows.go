//go:build windows

package decoder

import (
	"fmt"
	"syscall"
	"unsafe"
)

// mapGIW maps a .giw read-only on Windows, so its weights are file-backed as guardGIWFit assumes, where aikit's mmap.MapReadOnly
// reads the whole file onto the heap. The file is opened sharing read, write and delete, so a mapping does not stop a sidecar's
// replacement. Release with unmapGIW.
func mapGIW(path string) ([]byte, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	f, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer syscall.CloseHandle(f) // the mapping survives both handles
	var fi syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(f, &fi); err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	size := int64(fi.FileSizeHigh)<<32 | int64(fi.FileSizeLow)
	if size < 8 {
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, size)
	}
	section, err := syscall.CreateFileMapping(f, nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	defer syscall.CloseHandle(section)
	addr, err := syscall.MapViewOfFile(section, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return unsafe.Slice(*(**byte)(unsafe.Pointer(&addr)), size), nil
}

// unmapGIW releases what mapGIW returned.
func unmapGIW(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&data[0])))
}
