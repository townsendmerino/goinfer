// Package streamframe implements protocol stream framing, SSE transport, thinking-block splitting,
// reasoning-token budget resolution, /v1/responses reasoning items, and Responses/Anthropic stream
// state machines below internal/app/serveapp.
//
// It is a leaf over chat, internal/app/serveapp/carriedids, and internal/codec, with no dependency
// on decoder, engine, or server state.
package streamframe
