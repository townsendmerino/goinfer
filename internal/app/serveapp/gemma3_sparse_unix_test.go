//go:build !windows

package serveapp

import "os"

func markSparseForTest(*os.File) {}
