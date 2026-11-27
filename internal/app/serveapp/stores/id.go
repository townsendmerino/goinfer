package stores

import (
	"crypto/rand"
	"encoding/hex"
)

// MintID returns an opaque, unguessable 128-bit hex identifier for a response, message, file, batch, or tool call.
// Sequential or timestamp-seeded IDs would let a client enumerate other clients' stored state under a shared API key.
func MintID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("serveapp/stores: crypto/rand unavailable: " + err.Error())
	}
	return hex.EncodeToString(b[:])
}
