package multimodal

import "hash/fnv"

// HashImageBytes returns a content hash of raw image bytes, for the resident image-block reuse check (the imgHash parameter of
// decoder.GenerateVL / GenerateQwenVL). A plain non-cryptographic 64-bit hash is enough at agent-loop scale: there is no adversarial
// input, only "is this the same image byte-for-byte as the one already in the resident KV cache".
func HashImageBytes(image []byte) uint64 {
	h := fnv.New64a()
	_, _ = h.Write(image) // fnv's Write never returns an error
	return h.Sum64()
}
