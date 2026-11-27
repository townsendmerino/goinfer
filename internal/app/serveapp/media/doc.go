// Package media decodes inline base64 multimodal inputs for chat, responses and
// embeddings endpoints, and rewrites earlier conversation turns' media parts to
// visible omission notes when a single-span vision or audio pipeline keeps only
// the newest turn's media.
package media
