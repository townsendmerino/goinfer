package decoder

import "github.com/townsendmerino/aikit/mmap"

// unmapForTest releases what mapGIW returned.
func unmapForTest(data []byte) { _ = mmap.Unmap(data) }
